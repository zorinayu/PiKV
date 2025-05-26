#!/usr/bin/env python3
"""
PiKV CUDA Kernels Test Suite
Tests routing, compression, and scheduling kernels
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available, using PyTorch fallback")

class PiKVKernelTester:
    """Test suite for PiKV CUDA kernels"""
    
    def __init__(self, cuda_lib_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if cuda_lib_path is None:
            cuda_lib_path = os.path.join(os.path.dirname(__file__), 'libpikv_kernels.so')
        
        self.cuda_enabled = CUPY_AVAILABLE and os.path.exists(cuda_lib_path)
        
        if self.cuda_enabled:
            try:
                # Load CUDA library
                self.lib = cp.cuda.runtime.linkModule(cuda_lib_path)
                print(f"✓ Successfully loaded CUDA kernels: {cuda_lib_path}")
            except Exception as e:
                print(f"✗ Failed to load CUDA kernels: {e}")
                self.cuda_enabled = False
        else:
            print("✗ CUDA kernels not available, using PyTorch fallback")
            self.lib = None
    
    def test_routing_kernels(self) -> Dict[str, float]:
        """Test routing kernels performance"""
        print("\n" + "="*60)
        print("Testing Routing Kernels")
        print("="*60)
        
        results = {}
        
        # Test parameters
        batch_sizes = [16, 32, 64, 128]
        num_experts = 32
        k = 4
        temperature = 1.0
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch_size={batch_size}, num_experts={num_experts}, k={k}")
            
            # Generate test data
            routing_logits = torch.randn(batch_size, num_experts, device=self.device)
            
            if self.cuda_enabled:
                # Test CUDA kernel
                start_time = time.time()
                top_k_values, top_k_indices = self._cuda_topk_routing(
                    routing_logits, k, temperature
                )
                cuda_time = time.time() - start_time
                
                # Verify results
                torch_values, torch_indices = torch.topk(
                    torch.softmax(routing_logits / temperature, dim=1), k, dim=1
                )
                
                # Check correctness (values should be close)
                value_error = torch.mean(torch.abs(top_k_values - torch_values)).item()
                print(f"  CUDA kernel time: {cuda_time*1000:.2f} ms")
                print(f"  Value error vs PyTorch: {value_error:.6f}")
                
                results[f'routing_cuda_{batch_size}'] = cuda_time * 1000
            
            # Test PyTorch baseline
            start_time = time.time()
            torch_values, torch_indices = torch.topk(
                torch.softmax(routing_logits / temperature, dim=1), k, dim=1
            )
            torch_time = time.time() - start_time
            
            print(f"  PyTorch time: {torch_time*1000:.2f} ms")
            results[f'routing_torch_{batch_size}'] = torch_time * 1000
            
            if self.cuda_enabled:
                speedup = torch_time / cuda_time
                print(f"  Speedup: {speedup:.2f}x")
        
        return results
    
    def test_compression_kernels(self) -> Dict[str, float]:
        """Test compression kernels performance"""
        print("\n" + "="*60)
        print("Testing Compression Kernels")
        print("="*60)
        
        results = {}
        
        # Test parameters
        batch_sizes = [16, 32, 64]
        hidden_size = 256
        rank = 32
        alpha = 1.0
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch_size={batch_size}, hidden_size={hidden_size}")
            
            # Generate test data
            keys = torch.randn(batch_size, hidden_size, device=self.device)
            values = torch.randn(batch_size, hidden_size, device=self.device)
            importance = torch.rand(batch_size, device=self.device)
            
            # Test LoRA compression
            if self.cuda_enabled:
                start_time = time.time()
                compressed_keys, compressed_values = self._cuda_lora_compression(
                    keys, values, rank, alpha
                )
                cuda_time = time.time() - start_time
                
                print(f"  LoRA CUDA time: {cuda_time*1000:.2f} ms")
                results[f'lora_cuda_{batch_size}'] = cuda_time * 1000
            
            # Test quantization compression
            if self.cuda_enabled:
                start_time = time.time()
                quant_keys, quant_values, quant_params = self._cuda_quantization_compression(
                    keys, values, bits=8
                )
                cuda_time = time.time() - start_time
                
                # Calculate compression error
                key_error = torch.mean(torch.abs(keys - quant_keys)).item()
                value_error = torch.mean(torch.abs(values - quant_values)).item()
                
                print(f"  Quantization CUDA time: {cuda_time*1000:.2f} ms")
                print(f"  Quantization error (keys): {key_error:.6f}")
                print(f"  Quantization error (values): {value_error:.6f}")
                results[f'quant_cuda_{batch_size}'] = cuda_time * 1000
        
        return results
    
    def test_scheduling_kernels(self) -> Dict[str, float]:
        """Test cache scheduling kernels performance"""
        print("\n" + "="*60)
        print("Testing Cache Scheduling Kernels")
        print("="*60)
        
        results = {}
        
        # Test parameters
        cache_sizes = [512, 1024, 2048]
        hidden_size = 256
        batch_size = 32
        current_timestamp = 1000
        
        for cache_size in cache_sizes:
            print(f"\nTesting cache_size={cache_size}, batch_size={batch_size}")
            
            # Generate test cache data
            cache_keys = torch.randn(cache_size, hidden_size, device=self.device)
            cache_values = torch.randn(cache_size, hidden_size, device=self.device)
            cache_timestamps = torch.randint(0, current_timestamp, (cache_size,), device=self.device)
            cache_valid = torch.rand(cache_size, device=self.device) > 0.2  # 80% valid
            
            # New data to insert
            new_keys = torch.randn(batch_size, hidden_size, device=self.device)
            new_values = torch.randn(batch_size, hidden_size, device=self.device)
            
            if self.cuda_enabled:
                # Test LRU policy
                start_time = time.time()
                eviction_indices, insertion_indices = self._cuda_lru_scheduling(
                    cache_keys, cache_values, cache_timestamps, cache_valid,
                    new_keys, new_values, current_timestamp
                )
                cuda_time = time.time() - start_time
                
                print(f"  LRU CUDA time: {cuda_time*1000:.2f} ms")
                print(f"  Evictions: {torch.sum(eviction_indices >= 0).item()}")
                results[f'lru_cuda_{cache_size}'] = cuda_time * 1000
                
                # Test QUEST policy
                cache_importance = torch.rand(cache_size, device=self.device)
                cache_quality = torch.rand(cache_size, device=self.device)
                new_importance = torch.rand(batch_size, device=self.device)
                new_quality = torch.rand(batch_size, device=self.device)
                
                start_time = time.time()
                quest_evictions, quest_insertions = self._cuda_quest_scheduling(
                    cache_keys, cache_values, cache_importance, cache_quality,
                    cache_timestamps, cache_valid, new_keys, new_values,
                    new_importance, new_quality, current_timestamp, quest_threshold=0.1
                )
                cuda_time = time.time() - start_time
                
                print(f"  QUEST CUDA time: {cuda_time*1000:.2f} ms")
                print(f"  QUEST evictions: {torch.sum(quest_evictions >= 0).item()}")
                results[f'quest_cuda_{cache_size}'] = cuda_time * 1000
        
        return results
    
    def _cuda_topk_routing(self, routing_logits: torch.Tensor, k: int, temperature: float):
        """CUDA TopK routing implementation"""
        batch_size, num_experts = routing_logits.shape
        
        # Allocate output tensors
        top_k_values = torch.zeros(batch_size, k, device=self.device)
        top_k_indices = torch.zeros(batch_size, k, dtype=torch.int32, device=self.device)
        routing_weights = torch.zeros(batch_size, num_experts, device=self.device)
        
        # Convert to CuPy arrays
        cp_logits = cp.asarray(routing_logits)
        cp_values = cp.asarray(top_k_values)
        cp_indices = cp.asarray(top_k_indices)
        cp_weights = cp.asarray(routing_weights)
        
        # Call CUDA kernel
        self.lib.launch_topk_routing_kernel(
            cp_logits.data.ptr, cp_values.data.ptr, cp_indices.data.ptr,
            cp_weights.data.ptr, batch_size, num_experts, k, temperature, 0
        )
        
        return top_k_values, top_k_indices
    
    def _cuda_lora_compression(self, keys: torch.Tensor, values: torch.Tensor, 
                              rank: int, alpha: float):
        """CUDA LoRA compression implementation"""
        batch_size, hidden_size = keys.shape
        
        # Create random LoRA parameters
        lora_A_keys = torch.randn(hidden_size, rank, device=self.device) * 0.1
        lora_B_keys = torch.zeros(rank, hidden_size, device=self.device)
        lora_A_values = torch.randn(hidden_size, rank, device=self.device) * 0.1
        lora_B_values = torch.zeros(rank, hidden_size, device=self.device)
        
        # Allocate output tensors
        compressed_keys = torch.zeros_like(keys)
        compressed_values = torch.zeros_like(values)
        
        # Convert to CuPy arrays
        cp_keys = cp.asarray(keys)
        cp_values = cp.asarray(values)
        cp_comp_keys = cp.asarray(compressed_keys)
        cp_comp_values = cp.asarray(compressed_values)
        cp_lora_A_k = cp.asarray(lora_A_keys)
        cp_lora_B_k = cp.asarray(lora_B_keys)
        cp_lora_A_v = cp.asarray(lora_A_values)
        cp_lora_B_v = cp.asarray(lora_B_values)
        
        # Call CUDA kernel
        self.lib.launch_lora_compression_kernel(
            cp_keys.data.ptr, cp_values.data.ptr, cp_comp_keys.data.ptr,
            cp_comp_values.data.ptr, cp_lora_A_k.data.ptr, cp_lora_B_k.data.ptr,
            cp_lora_A_v.data.ptr, cp_lora_B_v.data.ptr, batch_size, hidden_size,
            rank, alpha, 0
        )
        
        return compressed_keys, compressed_values
    
    def _cuda_quantization_compression(self, keys: torch.Tensor, values: torch.Tensor, bits: int):
        """CUDA quantization compression implementation"""
        batch_size, hidden_size = keys.shape
        
        # Allocate output tensors
        compressed_keys = torch.zeros_like(keys)
        compressed_values = torch.zeros_like(values)
        quantization_params = torch.zeros(batch_size, 4, device=self.device)
        
        # Convert to CuPy arrays
        cp_keys = cp.asarray(keys)
        cp_values = cp.asarray(values)
        cp_comp_keys = cp.asarray(compressed_keys)
        cp_comp_values = cp.asarray(compressed_values)
        cp_params = cp.asarray(quantization_params)
        
        # Call CUDA kernel
        self.lib.launch_quantization_compression_kernel(
            cp_keys.data.ptr, cp_values.data.ptr, cp_comp_keys.data.ptr,
            cp_comp_values.data.ptr, cp_params.data.ptr, batch_size, hidden_size,
            bits, 0
        )
        
        return compressed_keys, compressed_values, quantization_params
    
    def _cuda_lru_scheduling(self, cache_keys: torch.Tensor, cache_values: torch.Tensor,
                            cache_timestamps: torch.Tensor, cache_valid: torch.Tensor,
                            new_keys: torch.Tensor, new_values: torch.Tensor,
                            current_timestamp: int):
        """CUDA LRU scheduling implementation"""
        cache_size, hidden_size = cache_keys.shape
        batch_size = new_keys.shape[0]
        
        # Allocate output tensors
        eviction_indices = torch.full((batch_size,), -1, dtype=torch.int32, device=self.device)
        insertion_indices = torch.full((batch_size,), -1, dtype=torch.int32, device=self.device)
        
        # Convert to CuPy arrays
        cp_cache_keys = cp.asarray(cache_keys)
        cp_cache_values = cp.asarray(cache_values)
        cp_timestamps = cp.asarray(cache_timestamps)
        cp_valid = cp.asarray(cache_valid)
        cp_new_keys = cp.asarray(new_keys)
        cp_new_values = cp.asarray(new_values)
        cp_evictions = cp.asarray(eviction_indices)
        cp_insertions = cp.asarray(insertion_indices)
        
        # Call CUDA kernel
        self.lib.launch_lru_cache_management_kernel(
            cp_cache_keys.data.ptr, cp_cache_values.data.ptr, cp_timestamps.data.ptr,
            cp_valid.data.ptr, cp_new_keys.data.ptr, cp_new_values.data.ptr,
            cp_evictions.data.ptr, cp_insertions.data.ptr, cache_size, hidden_size,
            batch_size, current_timestamp, 0
        )
        
        return eviction_indices, insertion_indices
    
    def _cuda_quest_scheduling(self, cache_keys: torch.Tensor, cache_values: torch.Tensor,
                              cache_importance: torch.Tensor, cache_quality: torch.Tensor,
                              cache_timestamps: torch.Tensor, cache_valid: torch.Tensor,
                              new_keys: torch.Tensor, new_values: torch.Tensor,
                              new_importance: torch.Tensor, new_quality: torch.Tensor,
                              current_timestamp: int, quest_threshold: float):
        """CUDA QUEST scheduling implementation"""
        cache_size, hidden_size = cache_keys.shape
        batch_size = new_keys.shape[0]
        
        # Allocate output tensors
        eviction_indices = torch.full((batch_size,), -1, dtype=torch.int32, device=self.device)
        insertion_indices = torch.full((batch_size,), -1, dtype=torch.int32, device=self.device)
        
        # Convert to CuPy arrays
        cp_cache_keys = cp.asarray(cache_keys)
        cp_cache_values = cp.asarray(cache_values)
        cp_importance = cp.asarray(cache_importance)
        cp_quality = cp.asarray(cache_quality)
        cp_timestamps = cp.asarray(cache_timestamps)
        cp_valid = cp.asarray(cache_valid)
        cp_new_keys = cp.asarray(new_keys)
        cp_new_values = cp.asarray(new_values)
        cp_new_importance = cp.asarray(new_importance)
        cp_new_quality = cp.asarray(new_quality)
        cp_evictions = cp.asarray(eviction_indices)
        cp_insertions = cp.asarray(insertion_indices)
        
        # Call CUDA kernel
        self.lib.launch_quest_cache_management_kernel(
            cp_cache_keys.data.ptr, cp_cache_values.data.ptr, cp_importance.data.ptr,
            cp_quality.data.ptr, cp_timestamps.data.ptr, cp_valid.data.ptr,
            cp_new_keys.data.ptr, cp_new_values.data.ptr, cp_new_importance.data.ptr,
            cp_new_quality.data.ptr, cp_evictions.data.ptr, cp_insertions.data.ptr,
            cache_size, hidden_size, batch_size, current_timestamp, quest_threshold, 0
        )
        
        return eviction_indices, insertion_indices
    
    def run_comprehensive_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run comprehensive benchmark of all kernels"""
        print("\n" + "="*80)
        print("PiKV CUDA Kernels Comprehensive Benchmark")
        print("="*80)
        
        results = {}
        
        # Test all kernel categories
        results['routing'] = self.test_routing_kernels()
        results['compression'] = self.test_compression_kernels()
        results['scheduling'] = self.test_scheduling_kernels()
        
        return results
    
    def plot_results(self, results: Dict[str, Dict[str, float]], save_path: str = 'kernel_benchmark.png'):
        """Plot benchmark results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot routing results
        if 'routing' in results:
            routing_data = results['routing']
            batch_sizes = [16, 32, 64, 128]
            cuda_times = [routing_data.get(f'routing_cuda_{bs}', 0) for bs in batch_sizes]
            torch_times = [routing_data.get(f'routing_torch_{bs}', 0) for bs in batch_sizes]
            
            axes[0].plot(batch_sizes, cuda_times, 'b-o', label='CUDA')
            axes[0].plot(batch_sizes, torch_times, 'r-s', label='PyTorch')
            axes[0].set_xlabel('Batch Size')
            axes[0].set_ylabel('Time (ms)')
            axes[0].set_title('Routing Kernel Performance')
            axes[0].legend()
            axes[0].grid(True)
        
        # Plot compression results
        if 'compression' in results:
            comp_data = results['compression']
            batch_sizes = [16, 32, 64]
            lora_times = [comp_data.get(f'lora_cuda_{bs}', 0) for bs in batch_sizes]
            quant_times = [comp_data.get(f'quant_cuda_{bs}', 0) for bs in batch_sizes]
            
            axes[1].plot(batch_sizes, lora_times, 'g-o', label='LoRA')
            axes[1].plot(batch_sizes, quant_times, 'm-s', label='Quantization')
            axes[1].set_xlabel('Batch Size')
            axes[1].set_ylabel('Time (ms)')
            axes[1].set_title('Compression Kernel Performance')
            axes[1].legend()
            axes[1].grid(True)
        
        # Plot scheduling results
        if 'scheduling' in results:
            sched_data = results['scheduling']
            cache_sizes = [512, 1024, 2048]
            lru_times = [sched_data.get(f'lru_cuda_{cs}', 0) for cs in cache_sizes]
            quest_times = [sched_data.get(f'quest_cuda_{cs}', 0) for cs in cache_sizes]
            
            axes[2].plot(cache_sizes, lru_times, 'c-o', label='LRU')
            axes[2].plot(cache_sizes, quest_times, 'y-s', label='QUEST')
            axes[2].set_xlabel('Cache Size')
            axes[2].set_ylabel('Time (ms)')
            axes[2].set_title('Scheduling Kernel Performance')
            axes[2].legend()
            axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Benchmark results saved to: {save_path}")

def main():
    """Main test function"""
    print("PiKV CUDA Kernels Test Suite")
    print("="*50)
    
    # Initialize tester
    tester = PiKVKernelTester()
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available, exiting...")
        return
    
    print(f"✓ Using device: {tester.device}")
    print(f"✓ GPU: {torch.cuda.get_device_name()}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    
    # Run comprehensive benchmark
    results = tester.run_comprehensive_benchmark()
    
    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    
    for category, data in results.items():
        print(f"\n{category.upper()} Results:")
        for test_name, time_ms in data.items():
            print(f"  {test_name}: {time_ms:.2f} ms")
    
    # Plot results
    tester.plot_results(results)
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    main() 