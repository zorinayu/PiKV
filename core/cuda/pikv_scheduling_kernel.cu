#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

// Constants
#define MAX_CACHE_SIZE 8192
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define INVALID_INDEX -1

// Eviction policy enumeration
enum EvictionPolicy {
    LRU_POLICY = 0,      // Least Recently Used
    LFU_POLICY = 1,      // Least Frequently Used
    QUEST_POLICY = 2,    // Quality-aware Eviction with Streaming
    ADAPTIVE_POLICY = 3, // Adaptive policy based on access patterns
    IMPORTANCE_POLICY = 4, // Importance-based eviction
    HYBRID_POLICY = 5    // Hybrid of multiple policies
};

// Cache entry structure
struct CacheEntry {
    float* key_data;
    float* value_data;
    float importance;
    int timestamp;
    int access_count;
    float quality_score;
    int last_access;
    bool is_valid;
};

// Device utility functions
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

__device__ __forceinline__ void warp_reduce_min(float* val, int* idx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, *val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, *idx, offset);
        if (other_val < *val) {
            *val = other_val;
            *idx = other_idx;
        }
    }
}

__device__ __forceinline__ float compute_cache_utility(
    float importance,
    int access_count,
    int age,
    float quality_score,
    int policy_type
) {
    switch (policy_type) {
        case LRU_POLICY:
            return -age;  // Negative age for LRU (older = lower utility)
        case LFU_POLICY:
            return access_count;  // Higher access count = higher utility
        case QUEST_POLICY:
            return importance * quality_score / (age + 1.0f);
        case IMPORTANCE_POLICY:
            return importance;
        default:
            return importance * access_count / (age + 1.0f);
    }
}

// LRU Cache Management Kernel
__global__ void lru_cache_management_kernel(
    float* __restrict__ cache_keys,           // [cache_size, hidden_size]
    float* __restrict__ cache_values,         // [cache_size, hidden_size]
    int* __restrict__ cache_timestamps,       // [cache_size]
    bool* __restrict__ cache_valid,           // [cache_size]
    const float* __restrict__ new_keys,       // [batch_size, hidden_size]
    const float* __restrict__ new_values,     // [batch_size, hidden_size]
    int* __restrict__ eviction_indices,       // [batch_size]
    int* __restrict__ insertion_indices,      // [batch_size]
    const int cache_size,
    const int hidden_size,
    const int batch_size,
    const int current_timestamp
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    __shared__ int s_oldest_timestamp;
    __shared__ int s_oldest_index;
    __shared__ int s_empty_index;
    
    if (tid == 0) {
        s_oldest_timestamp = current_timestamp + 1;
        s_oldest_index = -1;
        s_empty_index = -1;
    }
    __syncthreads();
    
    // Find oldest entry or empty slot
    for (int i = tid; i < cache_size; i += blockDim.x) {
        if (!cache_valid[i]) {
            // Found empty slot
            atomicMin(&s_empty_index, i);
        } else if (cache_timestamps[i] < s_oldest_timestamp) {
            atomicExch(&s_oldest_timestamp, cache_timestamps[i]);
            atomicExch(&s_oldest_index, i);
        }
    }
    __syncthreads();
    
    // Determine insertion index
    int insert_idx = (s_empty_index != -1) ? s_empty_index : s_oldest_index;
    
    if (tid == 0) {
        eviction_indices[batch_idx] = (s_empty_index != -1) ? -1 : s_oldest_index;
        insertion_indices[batch_idx] = insert_idx;
    }
    
    // Insert new entry
    if (insert_idx != -1) {
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            cache_keys[insert_idx * hidden_size + h] = new_keys[batch_idx * hidden_size + h];
            cache_values[insert_idx * hidden_size + h] = new_values[batch_idx * hidden_size + h];
        }
        
        if (tid == 0) {
            cache_timestamps[insert_idx] = current_timestamp;
            cache_valid[insert_idx] = true;
        }
    }
}

// LFU Cache Management Kernel
__global__ void lfu_cache_management_kernel(
    float* __restrict__ cache_keys,           // [cache_size, hidden_size]
    float* __restrict__ cache_values,         // [cache_size, hidden_size]
    int* __restrict__ cache_access_counts,    // [cache_size]
    bool* __restrict__ cache_valid,           // [cache_size]
    const float* __restrict__ new_keys,       // [batch_size, hidden_size]
    const float* __restrict__ new_values,     // [batch_size, hidden_size]
    int* __restrict__ eviction_indices,       // [batch_size]
    int* __restrict__ insertion_indices,      // [batch_size]
    const int cache_size,
    const int hidden_size,
    const int batch_size
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    __shared__ int s_min_access_count;
    __shared__ int s_lfu_index;
    __shared__ int s_empty_index;
    
    if (tid == 0) {
        s_min_access_count = INT_MAX;
        s_lfu_index = -1;
        s_empty_index = -1;
    }
    __syncthreads();
    
    // Find least frequently used entry or empty slot
    for (int i = tid; i < cache_size; i += blockDim.x) {
        if (!cache_valid[i]) {
            atomicMin(&s_empty_index, i);
        } else if (cache_access_counts[i] < s_min_access_count) {
            atomicExch(&s_min_access_count, cache_access_counts[i]);
            atomicExch(&s_lfu_index, i);
        }
    }
    __syncthreads();
    
    // Determine insertion index
    int insert_idx = (s_empty_index != -1) ? s_empty_index : s_lfu_index;
    
    if (tid == 0) {
        eviction_indices[batch_idx] = (s_empty_index != -1) ? -1 : s_lfu_index;
        insertion_indices[batch_idx] = insert_idx;
    }
    
    // Insert new entry
    if (insert_idx != -1) {
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            cache_keys[insert_idx * hidden_size + h] = new_keys[batch_idx * hidden_size + h];
            cache_values[insert_idx * hidden_size + h] = new_values[batch_idx * hidden_size + h];
        }
        
        if (tid == 0) {
            cache_access_counts[insert_idx] = 1;  // Initialize access count
            cache_valid[insert_idx] = true;
        }
    }
}

// QUEST Cache Management Kernel
__global__ void quest_cache_management_kernel(
    float* __restrict__ cache_keys,           // [cache_size, hidden_size]
    float* __restrict__ cache_values,         // [cache_size, hidden_size]
    float* __restrict__ cache_importance,     // [cache_size]
    float* __restrict__ cache_quality,        // [cache_size]
    int* __restrict__ cache_timestamps,       // [cache_size]
    bool* __restrict__ cache_valid,           // [cache_size]
    const float* __restrict__ new_keys,       // [batch_size, hidden_size]
    const float* __restrict__ new_values,     // [batch_size, hidden_size]
    const float* __restrict__ new_importance, // [batch_size]
    const float* __restrict__ new_quality,    // [batch_size]
    int* __restrict__ eviction_indices,       // [batch_size]
    int* __restrict__ insertion_indices,      // [batch_size]
    const int cache_size,
    const int hidden_size,
    const int batch_size,
    const int current_timestamp,
    const float quest_threshold
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    __shared__ float s_min_utility;
    __shared__ int s_victim_index;
    __shared__ int s_empty_index;
    
    if (tid == 0) {
        s_min_utility = FLT_MAX;
        s_victim_index = -1;
        s_empty_index = -1;
    }
    __syncthreads();
    
    // Calculate QUEST utility for each cache entry
    for (int i = tid; i < cache_size; i += blockDim.x) {
        if (!cache_valid[i]) {
            atomicMin(&s_empty_index, i);
        } else {
            int age = current_timestamp - cache_timestamps[i];
            float utility = cache_importance[i] * cache_quality[i] / (age + 1.0f);
            
            if (utility < s_min_utility) {
                atomicExch(&s_min_utility, utility);
                atomicExch(&s_victim_index, i);
            }
        }
    }
    __syncthreads();
    
    // Check if new entry should be inserted
    float new_imp = new_importance[batch_idx];
    float new_qual = new_quality[batch_idx];
    float new_utility = new_imp * new_qual;
    
    int insert_idx = -1;
    if (s_empty_index != -1) {
        insert_idx = s_empty_index;
    } else if (new_utility > s_min_utility + quest_threshold) {
        insert_idx = s_victim_index;
    }
    
    if (tid == 0) {
        eviction_indices[batch_idx] = (insert_idx == s_victim_index) ? s_victim_index : -1;
        insertion_indices[batch_idx] = insert_idx;
    }
    
    // Insert new entry if beneficial
    if (insert_idx != -1) {
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            cache_keys[insert_idx * hidden_size + h] = new_keys[batch_idx * hidden_size + h];
            cache_values[insert_idx * hidden_size + h] = new_values[batch_idx * hidden_size + h];
        }
        
        if (tid == 0) {
            cache_importance[insert_idx] = new_imp;
            cache_quality[insert_idx] = new_qual;
            cache_timestamps[insert_idx] = current_timestamp;
            cache_valid[insert_idx] = true;
        }
    }
}

// Adaptive Cache Management Kernel
__global__ void adaptive_cache_management_kernel(
    float* __restrict__ cache_keys,           // [cache_size, hidden_size]
    float* __restrict__ cache_values,         // [cache_size, hidden_size]
    float* __restrict__ cache_importance,     // [cache_size]
    int* __restrict__ cache_access_counts,    // [cache_size]
    int* __restrict__ cache_timestamps,       // [cache_size]
    bool* __restrict__ cache_valid,           // [cache_size]
    const float* __restrict__ new_keys,       // [batch_size, hidden_size]
    const float* __restrict__ new_values,     // [batch_size, hidden_size]
    const float* __restrict__ new_importance, // [batch_size]
    int* __restrict__ eviction_indices,       // [batch_size]
    int* __restrict__ insertion_indices,      // [batch_size]
    float* __restrict__ policy_weights,       // [3] (LRU, LFU, Importance weights)
    const int cache_size,
    const int hidden_size,
    const int batch_size,
    const int current_timestamp
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    __shared__ float s_min_score;
    __shared__ int s_victim_index;
    __shared__ int s_empty_index;
    
    if (tid == 0) {
        s_min_score = FLT_MAX;
        s_victim_index = -1;
        s_empty_index = -1;
    }
    __syncthreads();
    
    // Calculate adaptive score combining multiple policies
    for (int i = tid; i < cache_size; i += blockDim.x) {
        if (!cache_valid[i]) {
            atomicMin(&s_empty_index, i);
        } else {
            int age = current_timestamp - cache_timestamps[i];
            
            // Normalize scores
            float lru_score = 1.0f / (age + 1.0f);  // Recency
            float lfu_score = cache_access_counts[i] / 100.0f;  // Frequency (normalized)
            float imp_score = cache_importance[i];  // Importance
            
            // Weighted combination
            float adaptive_score = policy_weights[0] * lru_score + 
                                 policy_weights[1] * lfu_score + 
                                 policy_weights[2] * imp_score;
            
            if (adaptive_score < s_min_score) {
                atomicExch(&s_min_score, adaptive_score);
                atomicExch(&s_victim_index, i);
            }
        }
    }
    __syncthreads();
    
    // Determine insertion index
    int insert_idx = (s_empty_index != -1) ? s_empty_index : s_victim_index;
    
    if (tid == 0) {
        eviction_indices[batch_idx] = (s_empty_index != -1) ? -1 : s_victim_index;
        insertion_indices[batch_idx] = insert_idx;
    }
    
    // Insert new entry
    if (insert_idx != -1) {
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            cache_keys[insert_idx * hidden_size + h] = new_keys[batch_idx * hidden_size + h];
            cache_values[insert_idx * hidden_size + h] = new_values[batch_idx * hidden_size + h];
        }
        
        if (tid == 0) {
            cache_importance[insert_idx] = new_importance[batch_idx];
            cache_access_counts[insert_idx] = 1;
            cache_timestamps[insert_idx] = current_timestamp;
            cache_valid[insert_idx] = true;
        }
    }
}

// Cache Access Update Kernel
__global__ void cache_access_update_kernel(
    int* __restrict__ cache_access_counts,    // [cache_size]
    int* __restrict__ cache_timestamps,       // [cache_size]
    const int* __restrict__ access_indices,   // [batch_size]
    const int batch_size,
    const int current_timestamp
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int cache_idx = access_indices[batch_idx];
    if (cache_idx >= 0) {
        atomicAdd(&cache_access_counts[cache_idx], 1);
        cache_timestamps[cache_idx] = current_timestamp;
    }
}

// Cache Compaction Kernel
__global__ void cache_compaction_kernel(
    float* __restrict__ cache_keys,           // [cache_size, hidden_size]
    float* __restrict__ cache_values,         // [cache_size, hidden_size]
    float* __restrict__ cache_importance,     // [cache_size]
    int* __restrict__ cache_access_counts,    // [cache_size]
    int* __restrict__ cache_timestamps,       // [cache_size]
    bool* __restrict__ cache_valid,           // [cache_size]
    int* __restrict__ compaction_map,         // [cache_size]
    int* __restrict__ new_cache_size,         // [1]
    const int cache_size,
    const int hidden_size
) {
    int tid = threadIdx.x;
    int cache_idx = blockIdx.x;
    
    if (cache_idx >= cache_size) return;
    
    __shared__ int s_valid_count;
    __shared__ int s_write_pos;
    
    if (tid == 0) {
        s_valid_count = 0;
        s_write_pos = 0;
    }
    __syncthreads();
    
    // Count valid entries
    if (cache_valid[cache_idx]) {
        atomicAdd(&s_valid_count, 1);
    }
    __syncthreads();
    
    // Compact valid entries
    if (cache_valid[cache_idx]) {
        int write_pos = atomicAdd(&s_write_pos, 1);
        compaction_map[cache_idx] = write_pos;
        
        // Move data to compacted position
        for (int h = tid; h < hidden_size; h += blockDim.x) {
            cache_keys[write_pos * hidden_size + h] = cache_keys[cache_idx * hidden_size + h];
            cache_values[write_pos * hidden_size + h] = cache_values[cache_idx * hidden_size + h];
        }
        
        if (tid == 0) {
            cache_importance[write_pos] = cache_importance[cache_idx];
            cache_access_counts[write_pos] = cache_access_counts[cache_idx];
            cache_timestamps[write_pos] = cache_timestamps[cache_idx];
            cache_valid[write_pos] = true;
        }
    } else {
        compaction_map[cache_idx] = -1;
    }
    
    // Update cache size
    if (cache_idx == 0 && tid == 0) {
        *new_cache_size = s_valid_count;
    }
}

// Cache Statistics Kernel
__global__ void cache_statistics_kernel(
    const bool* __restrict__ cache_valid,     // [cache_size]
    const float* __restrict__ cache_importance, // [cache_size]
    const int* __restrict__ cache_access_counts, // [cache_size]
    const int* __restrict__ cache_timestamps,  // [cache_size]
    float* __restrict__ stats,                 // [6] (hit_rate, avg_importance, avg_access, avg_age, utilization, fragmentation)
    const int cache_size,
    const int current_timestamp,
    const int total_accesses
) {
    int tid = threadIdx.x;
    
    __shared__ float s_stats[6];
    __shared__ int s_valid_count;
    
    if (tid == 0) {
        for (int i = 0; i < 6; i++) s_stats[i] = 0.0f;
        s_valid_count = 0;
    }
    __syncthreads();
    
    // Accumulate statistics
    for (int i = tid; i < cache_size; i += blockDim.x) {
        if (cache_valid[i]) {
            atomicAdd(&s_valid_count, 1);
            atomicAdd(&s_stats[1], cache_importance[i]);  // avg_importance
            atomicAdd(&s_stats[2], (float)cache_access_counts[i]);  // avg_access
            atomicAdd(&s_stats[3], (float)(current_timestamp - cache_timestamps[i]));  // avg_age
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        // Calculate final statistics
        float utilization = (float)s_valid_count / cache_size;
        float fragmentation = 1.0f - utilization;
        
        stats[0] = (total_accesses > 0) ? (float)s_valid_count / total_accesses : 0.0f;  // hit_rate
        stats[1] = (s_valid_count > 0) ? s_stats[1] / s_valid_count : 0.0f;  // avg_importance
        stats[2] = (s_valid_count > 0) ? s_stats[2] / s_valid_count : 0.0f;  // avg_access
        stats[3] = (s_valid_count > 0) ? s_stats[3] / s_valid_count : 0.0f;  // avg_age
        stats[4] = utilization;
        stats[5] = fragmentation;
    }
}

// C++ wrapper functions
extern "C" {

void launch_lru_cache_management_kernel(
    float* cache_keys,
    float* cache_values,
    int* cache_timestamps,
    bool* cache_valid,
    const float* new_keys,
    const float* new_values,
    int* eviction_indices,
    int* insertion_indices,
    int cache_size,
    int hidden_size,
    int batch_size,
    int current_timestamp,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    lru_cache_management_kernel<<<grid, block, 0, stream>>>(
        cache_keys, cache_values, cache_timestamps, cache_valid,
        new_keys, new_values, eviction_indices, insertion_indices,
        cache_size, hidden_size, batch_size, current_timestamp
    );
}

void launch_lfu_cache_management_kernel(
    float* cache_keys,
    float* cache_values,
    int* cache_access_counts,
    bool* cache_valid,
    const float* new_keys,
    const float* new_values,
    int* eviction_indices,
    int* insertion_indices,
    int cache_size,
    int hidden_size,
    int batch_size,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    lfu_cache_management_kernel<<<grid, block, 0, stream>>>(
        cache_keys, cache_values, cache_access_counts, cache_valid,
        new_keys, new_values, eviction_indices, insertion_indices,
        cache_size, hidden_size, batch_size
    );
}

void launch_quest_cache_management_kernel(
    float* cache_keys,
    float* cache_values,
    float* cache_importance,
    float* cache_quality,
    int* cache_timestamps,
    bool* cache_valid,
    const float* new_keys,
    const float* new_values,
    const float* new_importance,
    const float* new_quality,
    int* eviction_indices,
    int* insertion_indices,
    int cache_size,
    int hidden_size,
    int batch_size,
    int current_timestamp,
    float quest_threshold,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    quest_cache_management_kernel<<<grid, block, 0, stream>>>(
        cache_keys, cache_values, cache_importance, cache_quality,
        cache_timestamps, cache_valid, new_keys, new_values,
        new_importance, new_quality, eviction_indices, insertion_indices,
        cache_size, hidden_size, batch_size, current_timestamp, quest_threshold
    );
}

void launch_adaptive_cache_management_kernel(
    float* cache_keys,
    float* cache_values,
    float* cache_importance,
    int* cache_access_counts,
    int* cache_timestamps,
    bool* cache_valid,
    const float* new_keys,
    const float* new_values,
    const float* new_importance,
    int* eviction_indices,
    int* insertion_indices,
    float* policy_weights,
    int cache_size,
    int hidden_size,
    int batch_size,
    int current_timestamp,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    adaptive_cache_management_kernel<<<grid, block, 0, stream>>>(
        cache_keys, cache_values, cache_importance, cache_access_counts,
        cache_timestamps, cache_valid, new_keys, new_values, new_importance,
        eviction_indices, insertion_indices, policy_weights,
        cache_size, hidden_size, batch_size, current_timestamp
    );
}

void launch_cache_access_update_kernel(
    int* cache_access_counts,
    int* cache_timestamps,
    const int* access_indices,
    int batch_size,
    int current_timestamp,
    cudaStream_t stream
) {
    dim3 grid((batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    cache_access_update_kernel<<<grid, block, 0, stream>>>(
        cache_access_counts, cache_timestamps, access_indices,
        batch_size, current_timestamp
    );
}

void launch_cache_compaction_kernel(
    float* cache_keys,
    float* cache_values,
    float* cache_importance,
    int* cache_access_counts,
    int* cache_timestamps,
    bool* cache_valid,
    int* compaction_map,
    int* new_cache_size,
    int cache_size,
    int hidden_size,
    cudaStream_t stream
) {
    dim3 grid(cache_size);
    dim3 block(BLOCK_SIZE);
    
    cache_compaction_kernel<<<grid, block, 0, stream>>>(
        cache_keys, cache_values, cache_importance, cache_access_counts,
        cache_timestamps, cache_valid, compaction_map, new_cache_size,
        cache_size, hidden_size
    );
}

void launch_cache_statistics_kernel(
    const bool* cache_valid,
    const float* cache_importance,
    const int* cache_access_counts,
    const int* cache_timestamps,
    float* stats,
    int cache_size,
    int current_timestamp,
    int total_accesses,
    cudaStream_t stream
) {
    dim3 grid(1);
    dim3 block(BLOCK_SIZE);
    
    cache_statistics_kernel<<<grid, block, 0, stream>>>(
        cache_valid, cache_importance, cache_access_counts, cache_timestamps,
        stats, cache_size, current_timestamp, total_accesses
    );
}

} // extern "C" 