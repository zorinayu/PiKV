#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cooperative_groups.h>
#include <math.h>

namespace cg = cooperative_groups;

// Constants
#define MAX_HIDDEN_SIZE 1024
#define MAX_RANK 128
#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Compression mode enumeration
enum CompressionMode {
    NO_COMPRESSION = 0,
    LORA_COMPRESSION = 1,
    QUANTIZATION_8BIT = 2,
    QUANTIZATION_4BIT = 3,
    PYRAMID_COMPRESSION = 4,
    SVD_COMPRESSION = 5,
    HYBRID_COMPRESSION = 6
};

// Device utility functions
__device__ __forceinline__ float quantize_8bit(float x, float scale, float zero_point) {
    return fmaxf(0.0f, fminf(255.0f, roundf(x / scale + zero_point)));
}

__device__ __forceinline__ float dequantize_8bit(float q, float scale, float zero_point) {
    return (q - zero_point) * scale;
}

__device__ __forceinline__ float quantize_4bit(float x, float scale, float zero_point) {
    return fmaxf(0.0f, fminf(15.0f, roundf(x / scale + zero_point)));
}

__device__ __forceinline__ float dequantize_4bit(float q, float scale, float zero_point) {
    return (q - zero_point) * scale;
}

__device__ __forceinline__ void warp_reduce_sum(float* val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        *val += __shfl_down_sync(0xffffffff, *val, offset);
    }
}

__device__ __forceinline__ void warp_reduce_max(float* val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        *val = fmaxf(*val, __shfl_down_sync(0xffffffff, *val, offset));
    }
}

__device__ __forceinline__ void warp_reduce_min(float* val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        *val = fminf(*val, __shfl_down_sync(0xffffffff, *val, offset));
    }
}

// LoRA Compression Kernel
__global__ void lora_compression_kernel(
    const float* __restrict__ input_keys,      // [batch_size, hidden_size]
    const float* __restrict__ input_values,    // [batch_size, hidden_size]
    float* __restrict__ compressed_keys,       // [batch_size, hidden_size]
    float* __restrict__ compressed_values,     // [batch_size, hidden_size]
    const float* __restrict__ lora_A_keys,     // [hidden_size, rank]
    const float* __restrict__ lora_B_keys,     // [rank, hidden_size]
    const float* __restrict__ lora_A_values,   // [hidden_size, rank]
    const float* __restrict__ lora_B_values,   // [rank, hidden_size]
    const int batch_size,
    const int hidden_size,
    const int rank,
    const float alpha
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;
    
    // LoRA transformation for keys: K' = K + alpha * (K * A) * B
    float key_val = input_keys[batch_idx * hidden_size + hidden_idx];
    float key_delta = 0.0f;
    
    // Compute (K * A) for this hidden dimension
    for (int r = 0; r < rank; r++) {
        float temp = 0.0f;
        for (int h = 0; h < hidden_size; h++) {
            temp += input_keys[batch_idx * hidden_size + h] * lora_A_keys[h * rank + r];
        }
        // Multiply by B and accumulate
        key_delta += temp * lora_B_keys[r * hidden_size + hidden_idx];
    }
    
    compressed_keys[batch_idx * hidden_size + hidden_idx] = key_val + alpha * key_delta;
    
    // LoRA transformation for values: V' = V + alpha * (V * A) * B
    float value_val = input_values[batch_idx * hidden_size + hidden_idx];
    float value_delta = 0.0f;
    
    // Compute (V * A) for this hidden dimension
    for (int r = 0; r < rank; r++) {
        float temp = 0.0f;
        for (int h = 0; h < hidden_size; h++) {
            temp += input_values[batch_idx * hidden_size + h] * lora_A_values[h * rank + r];
        }
        // Multiply by B and accumulate
        value_delta += temp * lora_B_values[r * hidden_size + hidden_idx];
    }
    
    compressed_values[batch_idx * hidden_size + hidden_idx] = value_val + alpha * value_delta;
}

// Quantization Compression Kernel
__global__ void quantization_compression_kernel(
    const float* __restrict__ input_keys,      // [batch_size, hidden_size]
    const float* __restrict__ input_values,    // [batch_size, hidden_size]
    float* __restrict__ compressed_keys,       // [batch_size, hidden_size]
    float* __restrict__ compressed_values,     // [batch_size, hidden_size]
    float* __restrict__ quantization_params,   // [batch_size, 4] (k_scale, k_zero, v_scale, v_zero)
    const int batch_size,
    const int hidden_size,
    const int bits
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    __shared__ float s_k_min, s_k_max, s_v_min, s_v_max;
    
    // Find min/max for keys and values in this batch
    if (tid == 0) {
        s_k_min = s_k_max = input_keys[batch_idx * hidden_size];
        s_v_min = s_v_max = input_values[batch_idx * hidden_size];
        
        for (int h = 1; h < hidden_size; h++) {
            float k_val = input_keys[batch_idx * hidden_size + h];
            float v_val = input_values[batch_idx * hidden_size + h];
            
            s_k_min = fminf(s_k_min, k_val);
            s_k_max = fmaxf(s_k_max, k_val);
            s_v_min = fminf(s_v_min, v_val);
            s_v_max = fmaxf(s_v_max, v_val);
        }
    }
    __syncthreads();
    
    // Calculate quantization parameters
    float qmax = (bits == 8) ? 255.0f : 15.0f;
    float k_scale = (s_k_max - s_k_min) / qmax;
    float k_zero = -s_k_min / k_scale;
    float v_scale = (s_v_max - s_v_min) / qmax;
    float v_zero = -s_v_min / v_scale;
    
    // Avoid division by zero
    k_scale = fmaxf(k_scale, 1e-8f);
    v_scale = fmaxf(v_scale, 1e-8f);
    
    // Store quantization parameters
    if (tid == 0) {
        quantization_params[batch_idx * 4 + 0] = k_scale;
        quantization_params[batch_idx * 4 + 1] = k_zero;
        quantization_params[batch_idx * 4 + 2] = v_scale;
        quantization_params[batch_idx * 4 + 3] = v_zero;
    }
    
    // Quantize and dequantize keys and values
    for (int h = tid; h < hidden_size; h += blockDim.x) {
        float k_val = input_keys[batch_idx * hidden_size + h];
        float v_val = input_values[batch_idx * hidden_size + h];
        
        if (bits == 8) {
            float k_quant = quantize_8bit(k_val, k_scale, k_zero);
            float v_quant = quantize_8bit(v_val, v_scale, v_zero);
            
            compressed_keys[batch_idx * hidden_size + h] = dequantize_8bit(k_quant, k_scale, k_zero);
            compressed_values[batch_idx * hidden_size + h] = dequantize_8bit(v_quant, v_scale, v_zero);
        } else {
            float k_quant = quantize_4bit(k_val, k_scale, k_zero);
            float v_quant = quantize_4bit(v_val, v_scale, v_zero);
            
            compressed_keys[batch_idx * hidden_size + h] = dequantize_4bit(k_quant, k_scale, k_zero);
            compressed_values[batch_idx * hidden_size + h] = dequantize_4bit(v_quant, v_scale, v_zero);
        }
    }
}

// Pyramid Compression Kernel
__global__ void pyramid_compression_kernel(
    const float* __restrict__ input_keys,      // [batch_size, hidden_size]
    const float* __restrict__ input_values,    // [batch_size, hidden_size]
    const float* __restrict__ importance,      // [batch_size]
    float* __restrict__ compressed_keys,       // [batch_size, hidden_size]
    float* __restrict__ compressed_values,     // [batch_size, hidden_size]
    const float* __restrict__ pyramid_weights, // [num_levels, hidden_size, compressed_size]
    const int batch_size,
    const int hidden_size,
    const int num_levels,
    const int* level_sizes,                    // [num_levels]
    const float importance_threshold
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;
    
    float imp = importance[batch_idx];
    int compression_level = 0;
    
    // Determine compression level based on importance
    if (imp > importance_threshold * 0.8f) {
        compression_level = 0;  // Minimal compression
    } else if (imp > importance_threshold * 0.5f) {
        compression_level = 1;  // Medium compression
    } else {
        compression_level = min(num_levels - 1, 2);  // Maximum compression
    }
    
    // Apply pyramid compression
    float key_val = input_keys[batch_idx * hidden_size + hidden_idx];
    float value_val = input_values[batch_idx * hidden_size + hidden_idx];
    
    // Compress through multiple levels
    float compressed_key = key_val;
    float compressed_value = value_val;
    
    for (int level = 0; level <= compression_level; level++) {
        int level_size = level_sizes[level];
        
        // Apply compression transformation
        float temp_key = 0.0f;
        float temp_value = 0.0f;
        
        for (int i = 0; i < level_size; i++) {
            int weight_idx = level * hidden_size * MAX_RANK + hidden_idx * MAX_RANK + i;
            temp_key += compressed_key * pyramid_weights[weight_idx];
            temp_value += compressed_value * pyramid_weights[weight_idx];
        }
        
        // Apply activation (ReLU)
        compressed_key = fmaxf(0.0f, temp_key);
        compressed_value = fmaxf(0.0f, temp_value);
    }
    
    // Add residual connection
    compressed_keys[batch_idx * hidden_size + hidden_idx] = key_val + compressed_key * 0.1f;
    compressed_values[batch_idx * hidden_size + hidden_idx] = value_val + compressed_value * 0.1f;
}

// SVD Compression Kernel
__global__ void svd_compression_kernel(
    const float* __restrict__ input_keys,      // [batch_size, hidden_size]
    const float* __restrict__ input_values,    // [batch_size, hidden_size]
    float* __restrict__ compressed_keys,       // [batch_size, hidden_size]
    float* __restrict__ compressed_values,     // [batch_size, hidden_size]
    const float* __restrict__ U_keys,          // [hidden_size, rank]
    const float* __restrict__ S_keys,          // [rank]
    const float* __restrict__ Vt_keys,         // [rank, hidden_size]
    const float* __restrict__ U_values,        // [hidden_size, rank]
    const float* __restrict__ S_values,        // [rank]
    const float* __restrict__ Vt_values,       // [rank, hidden_size]
    const int batch_size,
    const int hidden_size,
    const int rank
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;
    
    // SVD reconstruction for keys: K' = U * S * V^T
    float reconstructed_key = 0.0f;
    for (int r = 0; r < rank; r++) {
        float temp = 0.0f;
        for (int h = 0; h < hidden_size; h++) {
            temp += input_keys[batch_idx * hidden_size + h] * U_keys[h * rank + r];
        }
        temp *= S_keys[r];
        reconstructed_key += temp * Vt_keys[r * hidden_size + hidden_idx];
    }
    
    // SVD reconstruction for values: V' = U * S * V^T
    float reconstructed_value = 0.0f;
    for (int r = 0; r < rank; r++) {
        float temp = 0.0f;
        for (int h = 0; h < hidden_size; h++) {
            temp += input_values[batch_idx * hidden_size + h] * U_values[h * rank + r];
        }
        temp *= S_values[r];
        reconstructed_value += temp * Vt_values[r * hidden_size + hidden_idx];
    }
    
    compressed_keys[batch_idx * hidden_size + hidden_idx] = reconstructed_key;
    compressed_values[batch_idx * hidden_size + hidden_idx] = reconstructed_value;
}

// Adaptive Compression Selection Kernel
__global__ void adaptive_compression_selection_kernel(
    const float* __restrict__ input_keys,      // [batch_size, hidden_size]
    const float* __restrict__ input_values,    // [batch_size, hidden_size]
    const float* __restrict__ importance,      // [batch_size]
    int* __restrict__ compression_modes,       // [batch_size]
    float* __restrict__ compression_ratios,    // [batch_size]
    const int batch_size,
    const int hidden_size,
    const float high_importance_threshold,
    const float low_importance_threshold
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    float imp = importance[batch_idx];
    
    // Calculate input variance as complexity measure
    float mean = 0.0f;
    for (int h = 0; h < hidden_size; h++) {
        mean += input_keys[batch_idx * hidden_size + h];
    }
    mean /= hidden_size;
    
    float variance = 0.0f;
    for (int h = 0; h < hidden_size; h++) {
        float diff = input_keys[batch_idx * hidden_size + h] - mean;
        variance += diff * diff;
    }
    variance /= hidden_size;
    
    // Select compression mode based on importance and complexity
    if (imp > high_importance_threshold) {
        if (variance > 1.0f) {
            compression_modes[batch_idx] = LORA_COMPRESSION;
            compression_ratios[batch_idx] = 0.8f;  // Light compression
        } else {
            compression_modes[batch_idx] = SVD_COMPRESSION;
            compression_ratios[batch_idx] = 0.7f;
        }
    } else if (imp > low_importance_threshold) {
        compression_modes[batch_idx] = PYRAMID_COMPRESSION;
        compression_ratios[batch_idx] = 0.5f;  // Medium compression
    } else {
        compression_modes[batch_idx] = QUANTIZATION_8BIT;
        compression_ratios[batch_idx] = 0.25f;  // Heavy compression
    }
}

// Hybrid Compression Kernel
__global__ void hybrid_compression_kernel(
    const float* __restrict__ input_keys,      // [batch_size, hidden_size]
    const float* __restrict__ input_values,    // [batch_size, hidden_size]
    const int* __restrict__ compression_modes, // [batch_size]
    const float* __restrict__ compression_ratios, // [batch_size]
    float* __restrict__ compressed_keys,       // [batch_size, hidden_size]
    float* __restrict__ compressed_values,     // [batch_size, hidden_size]
    const float* __restrict__ lora_params,     // LoRA parameters
    const float* __restrict__ quant_params,    // Quantization parameters
    const int batch_size,
    const int hidden_size
) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;
    
    int mode = compression_modes[batch_idx];
    float ratio = compression_ratios[batch_idx];
    
    float key_val = input_keys[batch_idx * hidden_size + hidden_idx];
    float value_val = input_values[batch_idx * hidden_size + hidden_idx];
    
    float compressed_key = key_val;
    float compressed_value = value_val;
    
    // Apply compression based on selected mode
    switch (mode) {
        case LORA_COMPRESSION:
            // Apply LoRA with adaptive ratio
            compressed_key = key_val * (1.0f - ratio) + key_val * ratio;
            compressed_value = value_val * (1.0f - ratio) + value_val * ratio;
            break;
            
        case QUANTIZATION_8BIT:
            // Apply quantization
            compressed_key = roundf(key_val * 255.0f) / 255.0f;
            compressed_value = roundf(value_val * 255.0f) / 255.0f;
            break;
            
        case PYRAMID_COMPRESSION:
            // Apply pyramid compression with ratio
            compressed_key = key_val * ratio;
            compressed_value = value_val * ratio;
            break;
            
        default:
            // No compression
            break;
    }
    
    compressed_keys[batch_idx * hidden_size + hidden_idx] = compressed_key;
    compressed_values[batch_idx * hidden_size + hidden_idx] = compressed_value;
}

// Compression Quality Assessment Kernel
__global__ void compression_quality_assessment_kernel(
    const float* __restrict__ original_keys,    // [batch_size, hidden_size]
    const float* __restrict__ original_values,  // [batch_size, hidden_size]
    const float* __restrict__ compressed_keys,  // [batch_size, hidden_size]
    const float* __restrict__ compressed_values, // [batch_size, hidden_size]
    float* __restrict__ quality_scores,         // [batch_size]
    float* __restrict__ compression_errors,     // [batch_size]
    const int batch_size,
    const int hidden_size
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    __shared__ float s_key_error, s_value_error;
    __shared__ float s_key_norm, s_value_norm;
    
    if (tid == 0) {
        s_key_error = s_value_error = 0.0f;
        s_key_norm = s_value_norm = 0.0f;
    }
    __syncthreads();
    
    // Calculate MSE and norms
    for (int h = tid; h < hidden_size; h += blockDim.x) {
        float k_orig = original_keys[batch_idx * hidden_size + h];
        float k_comp = compressed_keys[batch_idx * hidden_size + h];
        float v_orig = original_values[batch_idx * hidden_size + h];
        float v_comp = compressed_values[batch_idx * hidden_size + h];
        
        float k_diff = k_orig - k_comp;
        float v_diff = v_orig - v_comp;
        
        atomicAdd(&s_key_error, k_diff * k_diff);
        atomicAdd(&s_value_error, v_diff * v_diff);
        atomicAdd(&s_key_norm, k_orig * k_orig);
        atomicAdd(&s_value_norm, v_orig * v_orig);
    }
    __syncthreads();
    
    if (tid == 0) {
        float total_error = (s_key_error + s_value_error) / (2.0f * hidden_size);
        float total_norm = (s_key_norm + s_value_norm) / (2.0f * hidden_size);
        
        // Normalized error
        float normalized_error = total_error / (total_norm + 1e-8f);
        
        // Quality score (higher is better)
        float quality = expf(-normalized_error);
        
        quality_scores[batch_idx] = quality;
        compression_errors[batch_idx] = normalized_error;
    }
}

// C++ wrapper functions
extern "C" {

void launch_lora_compression_kernel(
    const float* input_keys,
    const float* input_values,
    float* compressed_keys,
    float* compressed_values,
    const float* lora_A_keys,
    const float* lora_B_keys,
    const float* lora_A_values,
    const float* lora_B_values,
    int batch_size,
    int hidden_size,
    int rank,
    float alpha,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(min(hidden_size, BLOCK_SIZE));
    
    lora_compression_kernel<<<grid, block, 0, stream>>>(
        input_keys, input_values, compressed_keys, compressed_values,
        lora_A_keys, lora_B_keys, lora_A_values, lora_B_values,
        batch_size, hidden_size, rank, alpha
    );
}

void launch_quantization_compression_kernel(
    const float* input_keys,
    const float* input_values,
    float* compressed_keys,
    float* compressed_values,
    float* quantization_params,
    int batch_size,
    int hidden_size,
    int bits,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    quantization_compression_kernel<<<grid, block, 0, stream>>>(
        input_keys, input_values, compressed_keys, compressed_values,
        quantization_params, batch_size, hidden_size, bits
    );
}

void launch_pyramid_compression_kernel(
    const float* input_keys,
    const float* input_values,
    const float* importance,
    float* compressed_keys,
    float* compressed_values,
    const float* pyramid_weights,
    int batch_size,
    int hidden_size,
    int num_levels,
    const int* level_sizes,
    float importance_threshold,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(min(hidden_size, BLOCK_SIZE));
    
    pyramid_compression_kernel<<<grid, block, 0, stream>>>(
        input_keys, input_values, importance, compressed_keys, compressed_values,
        pyramid_weights, batch_size, hidden_size, num_levels, level_sizes, importance_threshold
    );
}

void launch_svd_compression_kernel(
    const float* input_keys,
    const float* input_values,
    float* compressed_keys,
    float* compressed_values,
    const float* U_keys,
    const float* S_keys,
    const float* Vt_keys,
    const float* U_values,
    const float* S_values,
    const float* Vt_values,
    int batch_size,
    int hidden_size,
    int rank,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(min(hidden_size, BLOCK_SIZE));
    
    svd_compression_kernel<<<grid, block, 0, stream>>>(
        input_keys, input_values, compressed_keys, compressed_values,
        U_keys, S_keys, Vt_keys, U_values, S_values, Vt_values,
        batch_size, hidden_size, rank
    );
}

void launch_adaptive_compression_selection_kernel(
    const float* input_keys,
    const float* input_values,
    const float* importance,
    int* compression_modes,
    float* compression_ratios,
    int batch_size,
    int hidden_size,
    float high_importance_threshold,
    float low_importance_threshold,
    cudaStream_t stream
) {
    dim3 grid((batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    adaptive_compression_selection_kernel<<<grid, block, 0, stream>>>(
        input_keys, input_values, importance, compression_modes, compression_ratios,
        batch_size, hidden_size, high_importance_threshold, low_importance_threshold
    );
}

void launch_hybrid_compression_kernel(
    const float* input_keys,
    const float* input_values,
    const int* compression_modes,
    const float* compression_ratios,
    float* compressed_keys,
    float* compressed_values,
    const float* lora_params,
    const float* quant_params,
    int batch_size,
    int hidden_size,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(min(hidden_size, BLOCK_SIZE));
    
    hybrid_compression_kernel<<<grid, block, 0, stream>>>(
        input_keys, input_values, compression_modes, compression_ratios,
        compressed_keys, compressed_values, lora_params, quant_params,
        batch_size, hidden_size
    );
}

void launch_compression_quality_assessment_kernel(
    const float* original_keys,
    const float* original_values,
    const float* compressed_keys,
    const float* compressed_values,
    float* quality_scores,
    float* compression_errors,
    int batch_size,
    int hidden_size,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(BLOCK_SIZE);
    
    compression_quality_assessment_kernel<<<grid, block, 0, stream>>>(
        original_keys, original_values, compressed_keys, compressed_values,
        quality_scores, compression_errors, batch_size, hidden_size
    );
}

} // extern "C" 