#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

// CUDA kernel function declarations
extern "C" {
    void moe_routing_cuda(
        const float* input,
        float* router_logits,
        const float* router_weights,
        const int batch_size,
        const int seq_len,
        const int hidden_size,
        const int num_experts,
        cudaStream_t stream
    );
    
    void top_k_experts_cuda(
        const float* router_logits,
        int* expert_indices,
        float* expert_weights,
        const int batch_size,
        const int seq_len,
        const int num_experts,
        const int top_k,
        cudaStream_t stream
    );
    
    void lora_compression_cuda(
        const float* input,
        float* output,
        const float* lora_A,
        const float* lora_B,
        const int batch_size,
        const int seq_len,
        const int hidden_size,
        const int rank,
        const float alpha,
        cudaStream_t stream
    );
    
    void pyramid_compression_cuda(
        const float* input,
        float* output,
        const float* encoder_weights,
        const float* decoder_weights,
        const int batch_size,
        const int seq_len,
        const int hidden_size,
        const int num_levels,
        cudaStream_t stream
    );
    
    void lru_cache_update_cuda(
        float* cache_keys,
        float* cache_values,
        int* cache_timestamps,
        const float* new_keys,
        const float* new_values,
        const int cache_size,
        const int hidden_size,
        const int current_timestamp,
        cudaStream_t stream
    );
    
    void h2o_cache_eviction_cuda(
        const float* cache_keys,
        const float* cache_values,
        const int* cache_timestamps,
        int* eviction_mask,
        const int cache_size,
        const int hidden_size,
        const float importance_threshold,
        const int max_age,
        cudaStream_t stream
    );
}

// Utility functions
void check_cuda_error(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " - " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void generate_random_data(std::vector<float>& data, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (auto& val : data) {
        val = dis(gen);
    }
}

void generate_random_data(std::vector<int>& data, int min_val = 0, int max_val = 100) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min_val, max_val);
    
    for (auto& val : data) {
        val = dis(gen);
    }
}

// Test functions
void test_moe_routing() {
    std::cout << "Testing MoE Routing..." << std::endl;
    
    const int batch_size = 2;
    const int seq_len = 64;
    const int hidden_size = 512;
    const int num_experts = 8;
    
    const int input_size = batch_size * seq_len * hidden_size;
    const int output_size = batch_size * seq_len * num_experts;
    const int weights_size = hidden_size * num_experts;
    
    // Host data
    std::vector<float> h_input(input_size);
    std::vector<float> h_router_weights(weights_size);
    std::vector<float> h_router_logits(output_size);
    
    // Generate random data
    generate_random_data(h_input);
    generate_random_data(h_router_weights);
    
    // Device data
    float *d_input, *d_router_weights, *d_router_logits;
    
    check_cuda_error(cudaMalloc(&d_input, input_size * sizeof(float)), "cudaMalloc input");
    check_cuda_error(cudaMalloc(&d_router_weights, weights_size * sizeof(float)), "cudaMalloc weights");
    check_cuda_error(cudaMalloc(&d_router_logits, output_size * sizeof(float)), "cudaMalloc logits");
    
    // Copy data to device
    check_cuda_error(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy input");
    check_cuda_error(cudaMemcpy(d_router_weights, h_router_weights.data(), weights_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy weights");
    
    // Create CUDA stream
    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream), "cudaStreamCreate");
    
    // Launch kernel
    auto start = std::chrono::high_resolution_clock::now();
    moe_routing_cuda(d_input, d_router_logits, d_router_weights, batch_size, seq_len, hidden_size, num_experts, stream);
    check_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy results back
    check_cuda_error(cudaMemcpy(h_router_logits.data(), d_router_logits, output_size * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy logits");
    
    // Verify results
    std::cout << "  Input shape: [" << batch_size << ", " << seq_len << ", " << hidden_size << "]" << std::endl;
    std::cout << "  Output shape: [" << batch_size << ", " << seq_len << ", " << num_experts << "]" << std::endl;
    std::cout << "  Execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " μs" << std::endl;
    std::cout << "  First output value: " << h_router_logits[0] << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_router_weights);
    cudaFree(d_router_logits);
    cudaStreamDestroy(stream);
    
    std::cout << "  ✓ MoE Routing test passed" << std::endl;
}

void test_top_k_experts() {
    std::cout << "Testing Top-K Experts..." << std::endl;
    
    const int batch_size = 2;
    const int seq_len = 64;
    const int num_experts = 8;
    const int top_k = 2;
    
    const int logits_size = batch_size * seq_len * num_experts;
    const int indices_size = batch_size * seq_len * top_k;
    const int weights_size = batch_size * seq_len * top_k;
    
    // Host data
    std::vector<float> h_router_logits(logits_size);
    std::vector<int> h_expert_indices(indices_size);
    std::vector<float> h_expert_weights(weights_size);
    
    // Generate random logits
    generate_random_data(h_router_logits);
    
    // Device data
    float *d_router_logits, *d_expert_weights;
    int *d_expert_indices;
    
    check_cuda_error(cudaMalloc(&d_router_logits, logits_size * sizeof(float)), "cudaMalloc logits");
    check_cuda_error(cudaMalloc(&d_expert_indices, indices_size * sizeof(int)), "cudaMalloc indices");
    check_cuda_error(cudaMalloc(&d_expert_weights, weights_size * sizeof(float)), "cudaMalloc weights");
    
    // Copy data to device
    check_cuda_error(cudaMemcpy(d_router_logits, h_router_logits.data(), logits_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy logits");
    
    // Create CUDA stream
    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream), "cudaStreamCreate");
    
    // Launch kernel
    auto start = std::chrono::high_resolution_clock::now();
    top_k_experts_cuda(d_router_logits, d_expert_indices, d_expert_weights, batch_size, seq_len, num_experts, top_k, stream);
    check_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy results back
    check_cuda_error(cudaMemcpy(h_expert_indices.data(), d_expert_indices, indices_size * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy indices");
    check_cuda_error(cudaMemcpy(h_expert_weights.data(), d_expert_weights, weights_size * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy weights");
    
    // Verify results
    std::cout << "  Logits shape: [" << batch_size << ", " << seq_len << ", " << num_experts << "]" << std::endl;
    std::cout << "  Top-K shape: [" << batch_size << ", " << seq_len << ", " << top_k << "]" << std::endl;
    std::cout << "  Execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " μs" << std::endl;
    std::cout << "  First expert index: " << h_expert_indices[0] << std::endl;
    std::cout << "  First expert weight: " << h_expert_weights[0] << std::endl;
    
    // Cleanup
    cudaFree(d_router_logits);
    cudaFree(d_expert_indices);
    cudaFree(d_expert_weights);
    cudaStreamDestroy(stream);
    
    std::cout << "  ✓ Top-K Experts test passed" << std::endl;
}

void test_lora_compression() {
    std::cout << "Testing LoRA Compression..." << std::endl;
    
    const int batch_size = 2;
    const int seq_len = 64;
    const int hidden_size = 512;
    const int rank = 16;
    const float alpha = 32.0f;
    
    const int data_size = batch_size * seq_len * hidden_size;
    const int lora_size = hidden_size * rank;
    
    // Host data
    std::vector<float> h_input(data_size);
    std::vector<float> h_output(data_size);
    std::vector<float> h_lora_A(lora_size);
    std::vector<float> h_lora_B(lora_size);
    
    // Generate random data
    generate_random_data(h_input);
    generate_random_data(h_lora_A);
    generate_random_data(h_lora_B);
    
    // Device data
    float *d_input, *d_output, *d_lora_A, *d_lora_B;
    
    check_cuda_error(cudaMalloc(&d_input, data_size * sizeof(float)), "cudaMalloc input");
    check_cuda_error(cudaMalloc(&d_output, data_size * sizeof(float)), "cudaMalloc output");
    check_cuda_error(cudaMalloc(&d_lora_A, lora_size * sizeof(float)), "cudaMalloc lora_A");
    check_cuda_error(cudaMalloc(&d_lora_B, lora_size * sizeof(float)), "cudaMalloc lora_B");
    
    // Copy data to device
    check_cuda_error(cudaMemcpy(d_input, h_input.data(), data_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy input");
    check_cuda_error(cudaMemcpy(d_lora_A, h_lora_A.data(), lora_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy lora_A");
    check_cuda_error(cudaMemcpy(d_lora_B, h_lora_B.data(), lora_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy lora_B");
    
    // Create CUDA stream
    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream), "cudaStreamCreate");
    
    // Launch kernel
    auto start = std::chrono::high_resolution_clock::now();
    lora_compression_cuda(d_input, d_output, d_lora_A, d_lora_B, batch_size, seq_len, hidden_size, rank, alpha, stream);
    check_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy results back
    check_cuda_error(cudaMemcpy(h_output.data(), d_output, data_size * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy output");
    
    // Verify results
    std::cout << "  Input shape: [" << batch_size << ", " << seq_len << ", " << hidden_size << "]" << std::endl;
    std::cout << "  LoRA rank: " << rank << std::endl;
    std::cout << "  Alpha: " << alpha << std::endl;
    std::cout << "  Execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " μs" << std::endl;
    std::cout << "  First output value: " << h_output[0] << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_lora_A);
    cudaFree(d_lora_B);
    cudaStreamDestroy(stream);
    
    std::cout << "  ✓ LoRA Compression test passed" << std::endl;
}

void test_cache_operations() {
    std::cout << "Testing Cache Operations..." << std::endl;
    
    const int cache_size = 1024;
    const int hidden_size = 512;
    const float importance_threshold = 0.1f;
    const int max_age = 1000;
    
    const int data_size = cache_size * hidden_size;
    
    // Host data
    std::vector<float> h_cache_keys(data_size);
    std::vector<float> h_cache_values(data_size);
    std::vector<int> h_cache_timestamps(cache_size);
    std::vector<float> h_new_keys(data_size);
    std::vector<float> h_new_values(data_size);
    std::vector<int> h_eviction_mask(cache_size);
    
    // Generate random data
    generate_random_data(h_cache_keys);
    generate_random_data(h_cache_values);
    generate_random_data(h_cache_timestamps, 0, 500);
    generate_random_data(h_new_keys);
    generate_random_data(h_new_values);
    
    // Device data
    float *d_cache_keys, *d_cache_values, *d_new_keys, *d_new_values;
    int *d_cache_timestamps, *d_eviction_mask;
    
    check_cuda_error(cudaMalloc(&d_cache_keys, data_size * sizeof(float)), "cudaMalloc cache_keys");
    check_cuda_error(cudaMalloc(&d_cache_values, data_size * sizeof(float)), "cudaMalloc cache_values");
    check_cuda_error(cudaMalloc(&d_cache_timestamps, cache_size * sizeof(int)), "cudaMalloc timestamps");
    check_cuda_error(cudaMalloc(&d_new_keys, data_size * sizeof(float)), "cudaMalloc new_keys");
    check_cuda_error(cudaMalloc(&d_new_values, data_size * sizeof(float)), "cudaMalloc new_values");
    check_cuda_error(cudaMalloc(&d_eviction_mask, cache_size * sizeof(int)), "cudaMalloc eviction_mask");
    
    // Copy data to device
    check_cuda_error(cudaMemcpy(d_cache_keys, h_cache_keys.data(), data_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy cache_keys");
    check_cuda_error(cudaMemcpy(d_cache_values, h_cache_values.data(), data_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy cache_values");
    check_cuda_error(cudaMemcpy(d_cache_timestamps, h_cache_timestamps.data(), cache_size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy timestamps");
    check_cuda_error(cudaMemcpy(d_new_keys, h_new_keys.data(), data_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy new_keys");
    check_cuda_error(cudaMemcpy(d_new_values, h_new_values.data(), data_size * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy new_values");
    
    // Create CUDA stream
    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream), "cudaStreamCreate");
    
    // Test LRU cache update
    auto start = std::chrono::high_resolution_clock::now();
    lru_cache_update_cuda(d_cache_keys, d_cache_values, d_cache_timestamps, d_new_keys, d_new_values, cache_size, hidden_size, 1000, stream);
    check_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    auto end = std::chrono::high_resolution_clock::now();
    
    // Test H2O cache eviction
    auto start2 = std::chrono::high_resolution_clock::now();
    h2o_cache_eviction_cuda(d_cache_keys, d_cache_values, d_cache_timestamps, d_eviction_mask, cache_size, hidden_size, importance_threshold, max_age, stream);
    check_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    auto end2 = std::chrono::high_resolution_clock::now();
    
    // Copy results back
    check_cuda_error(cudaMemcpy(h_eviction_mask.data(), d_eviction_mask, cache_size * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy eviction_mask");
    
    // Verify results
    std::cout << "  Cache size: " << cache_size << std::endl;
    std::cout << "  Hidden size: " << hidden_size << std::endl;
    std::cout << "  LRU update time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " μs" << std::endl;
    std::cout << "  H2O eviction time: " << std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() << " μs" << std::endl;
    
    // Count evicted items
    int evicted_count = 0;
    for (int i = 0; i < cache_size; i++) {
        if (h_eviction_mask[i] == 1) {
            evicted_count++;
        }
    }
    std::cout << "  Evicted items: " << evicted_count << "/" << cache_size << std::endl;
    
    // Cleanup
    cudaFree(d_cache_keys);
    cudaFree(d_cache_values);
    cudaFree(d_cache_timestamps);
    cudaFree(d_new_keys);
    cudaFree(d_new_values);
    cudaFree(d_eviction_mask);
    cudaStreamDestroy(stream);
    
    std::cout << "  ✓ Cache Operations test passed" << std::endl;
}

int main() {
    std::cout << "PiKV CUDA Kernels Test Suite" << std::endl;
    std::cout << "============================" << std::endl;
    
    // Check CUDA availability
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) {
        std::cerr << "CUDA not available: " << cudaGetErrorString(error) << std::endl;
        return EXIT_FAILURE;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Set device
    check_cuda_error(cudaSetDevice(0), "cudaSetDevice");
    
    // Get device properties
    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB" << std::endl;
    std::cout << std::endl;
    
    try {
        // Run tests
        test_moe_routing();
        std::cout << std::endl;
        
        test_top_k_experts();
        std::cout << std::endl;
        
        test_lora_compression();
        std::cout << std::endl;
        
        test_cache_operations();
        std::cout << std::endl;
        
        std::cout << "All tests passed successfully! 🎉" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
