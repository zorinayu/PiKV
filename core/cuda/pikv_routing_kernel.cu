#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

// Constants
#define MAX_EXPERTS 64
#define MAX_TOP_K 8
#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Routing mode enumeration
enum RoutingMode {
    TOP_K_ROUTING = 0,
    BALANCED_ROUTING = 1,
    ADAPTIVE_ROUTING = 2,
    IMPORTANCE_ROUTING = 3
};

// Device functions for routing utilities
__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ void warp_reduce_sum(float* val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        *val += __shfl_down_sync(0xffffffff, *val, offset);
    }
}

__device__ __forceinline__ void warp_reduce_max(float* val, int* idx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, *val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, *idx, offset);
        if (other_val > *val) {
            *val = other_val;
            *idx = other_idx;
        }
    }
}

// TopK Routing Kernel
__global__ void topk_routing_kernel(
    const float* __restrict__ routing_logits,  // [batch_size, num_experts]
    float* __restrict__ top_k_values,          // [batch_size, k]
    int* __restrict__ top_k_indices,           // [batch_size, k]
    float* __restrict__ routing_weights,       // [batch_size, num_experts]
    const int batch_size,
    const int num_experts,
    const int k,
    const float temperature
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory for this block
    __shared__ float s_logits[MAX_EXPERTS];
    __shared__ float s_values[MAX_TOP_K];
    __shared__ int s_indices[MAX_TOP_K];
    __shared__ float s_sum;
    
    // Load logits into shared memory
    if (tid < num_experts) {
        s_logits[tid] = routing_logits[batch_idx * num_experts + tid] / temperature;
    } else {
        s_logits[tid] = -INFINITY;
    }
    __syncthreads();
    
    // Apply softmax temperature scaling and find max for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < num_experts; i += blockDim.x) {
        max_val = fmaxf(max_val, s_logits[i]);
    }
    
    // Reduce to find global max
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
    }
    if (tid == 0) s_sum = max_val;
    __syncthreads();
    
    // Compute exp(logits - max) and sum
    float sum = 0.0f;
    for (int i = tid; i < num_experts; i += blockDim.x) {
        s_logits[i] = expf(s_logits[i] - s_sum);
        sum += s_logits[i];
    }
    
    // Reduce sum
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    if (tid == 0) s_sum = sum;
    __syncthreads();
    
    // Normalize to get probabilities
    for (int i = tid; i < num_experts; i += blockDim.x) {
        s_logits[i] /= s_sum;
    }
    __syncthreads();
    
    // Find top-k using iterative selection
    for (int k_idx = 0; k_idx < k; k_idx++) {
        float max_prob = -1.0f;
        int max_expert = -1;
        
        // Find maximum among remaining experts
        for (int i = tid; i < num_experts; i += blockDim.x) {
            bool already_selected = false;
            for (int j = 0; j < k_idx; j++) {
                if (s_indices[j] == i) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && s_logits[i] > max_prob) {
                max_prob = s_logits[i];
                max_expert = i;
            }
        }
        
        // Reduce to find global maximum
        warp_reduce_max(&max_prob, &max_expert);
        
        if (tid == 0) {
            s_values[k_idx] = max_prob;
            s_indices[k_idx] = max_expert;
        }
        __syncthreads();
    }
    
    // Write results to global memory
    if (tid < k) {
        top_k_values[batch_idx * k + tid] = s_values[tid];
        top_k_indices[batch_idx * k + tid] = s_indices[tid];
    }
    
    // Write full routing weights
    if (tid < num_experts) {
        routing_weights[batch_idx * num_experts + tid] = s_logits[tid];
    }
}

// Load Balancing Kernel
__global__ void load_balancing_kernel(
    const float* __restrict__ routing_weights,  // [batch_size, num_experts]
    float* __restrict__ expert_loads,           // [num_experts]
    float* __restrict__ load_balance_loss,      // [1]
    const int batch_size,
    const int num_experts,
    const float balance_coefficient
) {
    int expert_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (expert_idx >= num_experts) return;
    
    // Calculate load for this expert
    float load = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        load += routing_weights[b * num_experts + expert_idx];
    }
    expert_loads[expert_idx] = load / batch_size;
    
    __syncthreads();
    
    // Calculate load balancing loss (only first thread)
    if (expert_idx == 0) {
        float mean_load = 1.0f / num_experts;  // Ideal uniform load
        float loss = 0.0f;
        
        for (int i = 0; i < num_experts; i++) {
            float deviation = expert_loads[i] - mean_load;
            loss += deviation * deviation;
        }
        
        *load_balance_loss = balance_coefficient * loss;
    }
}

// Adaptive Routing Kernel with Importance Scoring
__global__ void adaptive_routing_kernel(
    const float* __restrict__ hidden_states,    // [batch_size, seq_len, hidden_size]
    const float* __restrict__ importance_scores, // [batch_size, seq_len]
    float* __restrict__ routing_logits,         // [batch_size, seq_len, num_experts]
    float* __restrict__ adaptive_weights,       // [batch_size, seq_len, num_experts]
    const float* __restrict__ router_weights,   // [hidden_size, num_experts]
    const float* __restrict__ router_bias,      // [num_experts]
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const int num_experts,
    const float importance_threshold,
    const float temperature
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int expert_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || expert_idx >= num_experts) return;
    
    // Calculate routing logit for this expert
    float logit = router_bias[expert_idx];
    
    // Compute dot product with hidden state
    for (int h = 0; h < hidden_size; h++) {
        float hidden_val = hidden_states[batch_idx * seq_len * hidden_size + seq_idx * hidden_size + h];
        logit += hidden_val * router_weights[h * num_experts + expert_idx];
    }
    
    // Apply temperature scaling
    logit /= temperature;
    
    // Get importance score for this token
    float importance = importance_scores[batch_idx * seq_len + seq_idx];
    
    // Adaptive adjustment based on importance
    if (importance > importance_threshold) {
        // High importance tokens get more concentrated routing
        logit *= (1.0f + importance);
    } else {
        // Low importance tokens get more diffuse routing
        logit *= (0.5f + 0.5f * importance);
    }
    
    routing_logits[batch_idx * seq_len * num_experts + seq_idx * num_experts + expert_idx] = logit;
    
    __syncthreads();
    
    // Compute softmax across experts for this token
    __shared__ float s_max_logit;
    __shared__ float s_sum_exp;
    
    if (expert_idx == 0) {
        // Find max logit for numerical stability
        float max_logit = -INFINITY;
        for (int e = 0; e < num_experts; e++) {
            float curr_logit = routing_logits[batch_idx * seq_len * num_experts + seq_idx * num_experts + e];
            max_logit = fmaxf(max_logit, curr_logit);
        }
        s_max_logit = max_logit;
        
        // Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            float curr_logit = routing_logits[batch_idx * seq_len * num_experts + seq_idx * num_experts + e];
            sum_exp += expf(curr_logit - max_logit);
        }
        s_sum_exp = sum_exp;
    }
    __syncthreads();
    
    // Compute final adaptive weight
    float curr_logit = routing_logits[batch_idx * seq_len * num_experts + seq_idx * num_experts + expert_idx];
    float weight = expf(curr_logit - s_max_logit) / s_sum_exp;
    
    adaptive_weights[batch_idx * seq_len * num_experts + seq_idx * num_experts + expert_idx] = weight;
}

// Importance-based Expert Selection Kernel
__global__ void importance_expert_selection_kernel(
    const float* __restrict__ token_embeddings,  // [batch_size, seq_len, hidden_size]
    const float* __restrict__ importance_scores,  // [batch_size, seq_len]
    int* __restrict__ expert_assignments,         // [batch_size, seq_len]
    float* __restrict__ assignment_confidence,    // [batch_size, seq_len]
    const float* __restrict__ expert_centroids,   // [num_experts, hidden_size]
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const int num_experts,
    const float confidence_threshold
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    __shared__ float s_distances[MAX_EXPERTS];
    __shared__ float s_token_embedding[256];  // Assuming hidden_size <= 256
    
    // Load token embedding into shared memory
    if (tid < hidden_size) {
        s_token_embedding[tid] = token_embeddings[batch_idx * seq_len * hidden_size + seq_idx * hidden_size + tid];
    }
    __syncthreads();
    
    // Compute distance to each expert centroid
    if (tid < num_experts) {
        float distance = 0.0f;
        for (int h = 0; h < hidden_size; h++) {
            float diff = s_token_embedding[h] - expert_centroids[tid * hidden_size + h];
            distance += diff * diff;
        }
        s_distances[tid] = sqrtf(distance);
    } else {
        s_distances[tid] = INFINITY;
    }
    __syncthreads();
    
    // Find closest expert
    if (tid == 0) {
        int best_expert = 0;
        float min_distance = s_distances[0];
        
        for (int e = 1; e < num_experts; e++) {
            if (s_distances[e] < min_distance) {
                min_distance = s_distances[e];
                best_expert = e;
            }
        }
        
        // Calculate confidence based on distance and importance
        float importance = importance_scores[batch_idx * seq_len + seq_idx];
        float confidence = importance * expf(-min_distance);
        
        expert_assignments[batch_idx * seq_len + seq_idx] = best_expert;
        assignment_confidence[batch_idx * seq_len + seq_idx] = confidence;
    }
}

// Expert Capacity Management Kernel
__global__ void expert_capacity_kernel(
    const int* __restrict__ expert_assignments,    // [batch_size, seq_len]
    const float* __restrict__ assignment_confidence, // [batch_size, seq_len]
    int* __restrict__ expert_capacities,           // [num_experts]
    int* __restrict__ overflow_tokens,             // [batch_size, seq_len]
    const int batch_size,
    const int seq_len,
    const int num_experts,
    const int max_capacity_per_expert
) {
    int expert_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (expert_idx >= num_experts) return;
    
    __shared__ int s_capacity_count;
    __shared__ int s_token_indices[512];  // Assuming reasonable batch_size * seq_len
    __shared__ float s_confidences[512];
    
    if (tid == 0) {
        s_capacity_count = 0;
    }
    __syncthreads();
    
    // Collect all tokens assigned to this expert
    for (int b = 0; b < batch_size; b++) {
        for (int s = tid; s < seq_len; s += blockDim.x) {
            int token_idx = b * seq_len + s;
            if (expert_assignments[token_idx] == expert_idx) {
                int pos = atomicAdd(&s_capacity_count, 1);
                if (pos < 512) {  // Safety check
                    s_token_indices[pos] = token_idx;
                    s_confidences[pos] = assignment_confidence[token_idx];
                }
            }
        }
    }
    __syncthreads();
    
    // Sort tokens by confidence (simple bubble sort for small arrays)
    if (tid == 0) {
        for (int i = 0; i < s_capacity_count - 1; i++) {
            for (int j = 0; j < s_capacity_count - i - 1; j++) {
                if (s_confidences[j] < s_confidences[j + 1]) {
                    // Swap confidences
                    float temp_conf = s_confidences[j];
                    s_confidences[j] = s_confidences[j + 1];
                    s_confidences[j + 1] = temp_conf;
                    
                    // Swap indices
                    int temp_idx = s_token_indices[j];
                    s_token_indices[j] = s_token_indices[j + 1];
                    s_token_indices[j + 1] = temp_idx;
                }
            }
        }
        
        // Mark overflow tokens
        expert_capacities[expert_idx] = min(s_capacity_count, max_capacity_per_expert);
        
        for (int i = max_capacity_per_expert; i < s_capacity_count; i++) {
            overflow_tokens[s_token_indices[i]] = 1;  // Mark as overflow
        }
    }
}

// C++ wrapper functions
extern "C" {

void launch_topk_routing_kernel(
    const float* routing_logits,
    float* top_k_values,
    int* top_k_indices,
    float* routing_weights,
    int batch_size,
    int num_experts,
    int k,
    float temperature,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(min(num_experts, BLOCK_SIZE));
    
    topk_routing_kernel<<<grid, block, 0, stream>>>(
        routing_logits, top_k_values, top_k_indices, routing_weights,
        batch_size, num_experts, k, temperature
    );
}

void launch_load_balancing_kernel(
    const float* routing_weights,
    float* expert_loads,
    float* load_balance_loss,
    int batch_size,
    int num_experts,
    float balance_coefficient,
    cudaStream_t stream
) {
    dim3 grid((num_experts + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    load_balancing_kernel<<<grid, block, 0, stream>>>(
        routing_weights, expert_loads, load_balance_loss,
        batch_size, num_experts, balance_coefficient
    );
}

void launch_adaptive_routing_kernel(
    const float* hidden_states,
    const float* importance_scores,
    float* routing_logits,
    float* adaptive_weights,
    const float* router_weights,
    const float* router_bias,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_experts,
    float importance_threshold,
    float temperature,
    cudaStream_t stream
) {
    dim3 grid(batch_size, seq_len);
    dim3 block(num_experts);
    
    adaptive_routing_kernel<<<grid, block, 0, stream>>>(
        hidden_states, importance_scores, routing_logits, adaptive_weights,
        router_weights, router_bias, batch_size, seq_len, hidden_size,
        num_experts, importance_threshold, temperature
    );
}

void launch_importance_expert_selection_kernel(
    const float* token_embeddings,
    const float* importance_scores,
    int* expert_assignments,
    float* assignment_confidence,
    const float* expert_centroids,
    int batch_size,
    int seq_len,
    int hidden_size,
    int num_experts,
    float confidence_threshold,
    cudaStream_t stream
) {
    dim3 grid(batch_size, seq_len);
    dim3 block(max(num_experts, hidden_size));
    
    importance_expert_selection_kernel<<<grid, block, 0, stream>>>(
        token_embeddings, importance_scores, expert_assignments, assignment_confidence,
        expert_centroids, batch_size, seq_len, hidden_size, num_experts, confidence_threshold
    );
}

void launch_expert_capacity_kernel(
    const int* expert_assignments,
    const float* assignment_confidence,
    int* expert_capacities,
    int* overflow_tokens,
    int batch_size,
    int seq_len,
    int num_experts,
    int max_capacity_per_expert,
    cudaStream_t stream
) {
    dim3 grid(num_experts);
    dim3 block(BLOCK_SIZE);
    
    expert_capacity_kernel<<<grid, block, 0, stream>>>(
        expert_assignments, assignment_confidence, expert_capacities, overflow_tokens,
        batch_size, seq_len, num_experts, max_capacity_per_expert
    );
}

} // extern "C" 