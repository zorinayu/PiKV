import torch
import torch.nn as nn
import torch.distributed as dist
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from typing import List, Dict, Tuple, Union
import time
import psutil
import os
from tqdm import tqdm
import json
import numpy as np
from datasets import load_dataset
import torch.nn.functional as F

class CacheCompressionTest:
    def __init__(self, model_name: str, num_gpus: int = 8):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get model's hidden size and other parameters
        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Initialize compression schemes
        self.compression_schemes = {
            'no_compression': self._no_compression,
            'pikv': self._pikv_compression,
            'dynamic': self._dynamic_compression,
            'pyramid': self._pyramid_compression,
            'importance': self._importance_compression,
            'adaptive': self._adaptive_compression,
            'hybrid': self._hybrid_compression
        }
        
        # Initialize datasets
        self.datasets = {
            'qwen': self._load_qwen_dataset,
            'wikitext': self._load_wikitext_dataset,
            'pile': self._load_pile_dataset,
            'c4': self._load_c4_dataset,
            'bookcorpus': self._load_bookcorpus_dataset
        }
    
    def _no_compression(self, cache_size: int) -> Dict[str, torch.Tensor]:
        """No compression scheme - direct storage."""
        # Initialize tensors for direct storage
        keys = torch.zeros(cache_size, self.num_heads, self.head_dim, device=self.device)
        values = torch.zeros(cache_size, self.num_heads, self.head_dim, device=self.device)
        return {'keys': keys, 'values': values}
    
    def _pikv_compression(self, cache_size: int) -> Dict[str, torch.Tensor]:
        """PiKV compression scheme with importance-based compression."""
        compressed_size = cache_size // 2
        # Initialize tensors with importance scores
        keys = torch.zeros(compressed_size, self.num_heads, self.head_dim, device=self.device)
        values = torch.zeros(compressed_size, self.num_heads, self.head_dim, device=self.device)
        importance = torch.zeros(compressed_size, device=self.device)
        
        # Apply importance-based compression
        def compress(x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
            # Sort by importance
            _, indices = torch.sort(importance, descending=True)
            # Take top-k most important elements
            return x[indices[:compressed_size]]
        
        return {
            'keys': keys,
            'values': values,
            'importance': importance,
            'compress_fn': compress
        }
    
    def _dynamic_compression(self, cache_size: int) -> Dict[str, torch.Tensor]:
        """Dynamic compression scheme with adaptive compression ratio."""
        compressed_size = cache_size // 4
        # Initialize tensors with dynamic compression
        keys = torch.zeros(compressed_size, self.num_heads, self.head_dim, device=self.device)
        values = torch.zeros(compressed_size, self.num_heads, self.head_dim, device=self.device)
        compression_ratio = torch.tensor(0.25, device=self.device)
        
        # Apply dynamic compression based on input characteristics
        def compress(x: torch.Tensor) -> torch.Tensor:
            # Calculate compression ratio based on input variance
            variance = torch.var(x, dim=0)
            ratio = torch.sigmoid(variance.mean()) * compression_ratio
            # Apply compression
            return x[:int(x.size(0) * ratio)]
        
        return {
            'keys': keys,
            'values': values,
            'compression_ratio': compression_ratio,
            'compress_fn': compress
        }
    
    def _pyramid_compression(self, cache_size: int) -> Dict[str, torch.Tensor]:
        """Pyramid compression scheme with multiple levels."""
        levels = 3
        compressed_sizes = [cache_size // (2**i) for i in range(levels)]
        
        # Initialize tensors for each level
        keys = []
        values = []
        for size in compressed_sizes:
            keys.append(torch.zeros(size, self.num_heads, self.head_dim, device=self.device))
            values.append(torch.zeros(size, self.num_heads, self.head_dim, device=self.device))
        
        # Apply pyramid compression
        def compress(x: torch.Tensor) -> List[torch.Tensor]:
            compressed = []
            current = x
            for size in compressed_sizes:
                # Downsample using average pooling
                if current.size(0) > size:
                    current = F.avg_pool1d(current.permute(1, 2, 0), kernel_size=2).permute(2, 0, 1)
                compressed.append(current)
            return compressed
        
        return {
            'keys': keys,
            'values': values,
            'level_sizes': torch.tensor(compressed_sizes, device=self.device),
            'compress_fn': compress
        }
    
    def _importance_compression(self, cache_size: int) -> Dict[str, torch.Tensor]:
        """Importance-based compression scheme."""
        compressed_size = cache_size // 3
        # Initialize tensors with importance scores
        keys = torch.zeros(compressed_size, self.num_heads, self.head_dim, device=self.device)
        values = torch.zeros(compressed_size, self.num_heads, self.head_dim, device=self.device)
        importance_scores = torch.zeros(compressed_size, device=self.device)
        
        # Apply importance-based compression
        def compress(x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
            # Calculate importance scores
            scores = torch.norm(x, dim=(1, 2))
            # Sort by importance
            _, indices = torch.sort(scores, descending=True)
            # Take top-k most important elements
            return x[indices[:compressed_size]]
        
        return {
            'keys': keys,
            'values': values,
            'importance_scores': importance_scores,
            'compress_fn': compress
        }
    
    def _adaptive_compression(self, cache_size: int) -> Dict[str, torch.Tensor]:
        """Adaptive compression scheme with dynamic compression ratios."""
        compressed_size = cache_size // 2
        # Initialize tensors with adaptive compression
        keys = torch.zeros(compressed_size, self.num_heads, self.head_dim, device=self.device)
        values = torch.zeros(compressed_size, self.num_heads, self.head_dim, device=self.device)
        compression_ratios = torch.zeros(compressed_size, device=self.device)
        
        # Apply adaptive compression
        def compress(x: torch.Tensor) -> torch.Tensor:
            # Calculate compression ratios based on input characteristics
            ratios = torch.sigmoid(torch.norm(x, dim=(1, 2)))
            # Apply compression
            mask = ratios > 0.5
            return x[mask]
        
        return {
            'keys': keys,
            'values': values,
            'compression_ratios': compression_ratios,
            'compress_fn': compress
        }
    
    def _hybrid_compression(self, cache_size: int) -> Dict[str, torch.Tensor]:
        """Hybrid compression scheme combining multiple methods."""
        compressed_size = cache_size // 4
        # Initialize tensors for hybrid compression
        keys = torch.zeros(compressed_size, self.num_heads, self.head_dim, device=self.device)
        values = torch.zeros(compressed_size, self.num_heads, self.head_dim, device=self.device)
        importance = torch.zeros(compressed_size, device=self.device)
        compression_ratio = torch.tensor(0.25, device=self.device)
        
        # Apply hybrid compression
        def compress(x: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
            # Combine importance and dynamic compression
            importance_scores = torch.norm(x, dim=(1, 2))
            variance = torch.var(x, dim=0)
            combined_scores = importance_scores * torch.sigmoid(variance.mean())
            
            # Sort by combined scores
            _, indices = torch.sort(combined_scores, descending=True)
            # Apply compression ratio
            return x[indices[:int(x.size(0) * compression_ratio)]]
        
        return {
            'keys': keys,
            'values': values,
            'importance': importance,
            'compression_ratio': compression_ratio,
            'compress_fn': compress
        }
    
    def _load_qwen_dataset(self) -> List[str]:
        """Load Qwen dataset."""
        dataset = load_dataset("Qwen/Qwen2.5-7B-Instruct-1M")
        return [item['text'] for item in dataset['train']]
    
    def _load_wikitext_dataset(self) -> List[str]:
        """Load Wikitext dataset."""
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        return [item['text'] for item in dataset['train']]
    
    def _load_pile_dataset(self) -> List[str]:
        """Load Pile dataset."""
        dataset = load_dataset("EleutherAI/pile")
        return [item['text'] for item in dataset['train']]
    
    def _load_c4_dataset(self) -> List[str]:
        """Load C4 dataset."""
        dataset = load_dataset("c4", "en")
        return [item['text'] for item in dataset['train']]
    
    def _load_bookcorpus_dataset(self) -> List[str]:
        """Load BookCorpus dataset."""
        dataset = load_dataset("bookcorpus")
        return [item['text'] for item in dataset['train']]
    
    def _measure_memory_usage(self) -> float:
        """Measure current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024
    
    def _test_compression_scheme(self, scheme_name: str, cache_size: int, dataset_name: str) -> Dict:
        """Test a specific compression scheme."""
        # Initialize cache
        cache = self.compression_schemes[scheme_name](cache_size)
        
        # Load dataset
        dataset = self.datasets[dataset_name]()
        
        # Measure initial memory
        initial_memory = self._measure_memory_usage()
        
        # Process dataset
        start_time = time.time()
        max_tokens = 0
        for text in tqdm(dataset[:100], desc=f"Testing {scheme_name}"):
            # Tokenize text
            tokens = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
            
            # Update max tokens
            max_tokens = max(max_tokens, tokens.size(1))
            
            # Process through model
            with torch.no_grad():
                outputs = self.model(tokens)
        
        end_time = time.time()
        final_memory = self._measure_memory_usage()
        
        return {
            'scheme': scheme_name,
            'cache_size': cache_size,
            'dataset': dataset_name,
            'max_tokens': max_tokens,
            'memory_usage': final_memory - initial_memory,
            'processing_time': end_time - start_time
        }
    
    def run_tests(self, cache_sizes: List[int] = [1024, 2048, 4096, 8192, 16384, 32768]) -> Dict:
        """Run comprehensive tests."""
        results = {}
        
        for dataset_name in self.datasets:
            results[dataset_name] = {}
            for scheme_name in self.compression_schemes:
                results[dataset_name][scheme_name] = []
                for cache_size in cache_sizes:
                    result = self._test_compression_scheme(scheme_name, cache_size, dataset_name)
                    results[dataset_name][scheme_name].append(result)
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "compression_results.json"):
        """Save test results to file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    # Initialize test
    test = CacheCompressionTest("Qwen/Qwen2.5-7B-Instruct-1M")
    
    # Run tests
    results = test.run_tests()
    
    # Save results
    test.save_results(results)
    
    # Print summary
    print("\nTest Results Summary:")
    for dataset_name, dataset_results in results.items():
        print(f"\nDataset: {dataset_name}")
        for scheme_name, scheme_results in dataset_results.items():
            print(f"\nCompression Scheme: {scheme_name}")
            for result in scheme_results:
                print(f"Cache Size: {result['cache_size']}, Max Tokens: {result['max_tokens']}, "
                      f"Memory Usage: {result['memory_usage']:.2f}GB, "
                      f"Time: {result['processing_time']:.2f}s")

if __name__ == "__main__":
    main() 