import torch
import torch.nn as nn
import torch.distributed as dist
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from core.distributed.distributed_pikv import DistributedPiKVMoE
from core.distributed.config import config
from core.distributed.distributed_config import distributed_config as dconfig
from core.single.model_compression.matrix_defactorization import LoRACompressor, LoRAPlusCompressor
from core.single.model_compression.cache_reduction import PyramidCompressor, FastVCompressor
from core.single.model_compression.distillation import FastVideoCompressor, MiniLLMCompressor
from core.single.model_compression.compression_utils import CompressionEvaluator, SimpleNextTokenPredictor

def setup_distributed():
    """Initialize distributed environment with error handling."""
    try:
        if not dist.is_initialized():
            # Get rank and world_size from environment variables
            rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            print(f"Initializing distributed environment on rank {rank}, local_rank {local_rank}")
            
            # Initialize process group
            dist.init_process_group(
                backend=dconfig['dist_backend'],
                init_method='env://',  # Use environment variables
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=30)
            )
            print(f"Successfully initialized distributed environment on rank {rank}")
    except Exception as e:
        print(f"Error initializing distributed environment: {e}")
        raise

class DistributedPiKVCacheWithCompression:
    def __init__(
        self, 
        model_name: str = "gpt2", 
        max_length: int = 1024,
        compression_method: str = "lora",
        compression_ratio: float = 0.5
    ):
        # Initialize distributed environment
        setup_distributed()
        
        # Get actual rank and local_rank from environment
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        print(f"Process started with rank {self.rank}, local_rank {self.local_rank}, world_size {self.world_size}")
        
        # Initialize model and tokenizer
        print(f"Rank {self.rank}: Loading model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Get model's hidden size
        self.hidden_size = self.model.config.hidden_size
        
        # Update global config with model's hidden size
        config['hidden_size'] = self.hidden_size
        
        # Initialize Distributed PiKV MoE with higher LoRA rank
        print(f"Rank {self.rank}: Initializing DistributedPiKVMoE...")
        self.pikv = DistributedPiKVMoE(rank=8, alpha=1.0)
        
        # Initialize compressor based on specified method
        print(f"Rank {self.rank}: Initializing compressor ({compression_method})...")
        self.compressor = self._create_compressor(compression_method, compression_ratio)
        self.compression_method = compression_method
        self.compression_ratio = compression_ratio
        
        # Performance metrics
        self.compression_times = []
        self.decompression_times = []
        self.original_sizes = []
        self.compressed_sizes = []
        self.prediction_losses = []
        
        # Move model, PiKV, and compressor to device
        self.device = torch.device(f"cuda:{self.local_rank}")
        print(f"Rank {self.rank}: Moving model to device {self.device}")
        self.model = self.model.to(self.device)
        self.pikv = self.pikv.to(self.device)
        self.compressor = self.compressor.to(self.device)
        
        # Initialize KV cache
        self.kv_cache = {}
        self.current_length = 0
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        print(f"Rank {self.rank}: Initialization complete")
    
    def _create_compressor(self, method: str, compression_ratio: float) -> nn.Module:
        """Create compressor based on specified method."""
        if method.lower() == "lora":
            rank = max(int(self.hidden_size * compression_ratio), 1)
            return LoRACompressor(
                hidden_size=self.hidden_size,
                rank=rank,
                alpha=32.0
            )
        elif method.lower() == "lora+":
            ranks = [4, 8, 16]
            return LoRAPlusCompressor(
                hidden_size=self.hidden_size,
                ranks=ranks,
                alpha=32.0,
                importance_thresholds=[0.3, 0.7]
            )
        elif method.lower() == "pyramid":
            return PyramidCompressor(
                hidden_size=self.hidden_size,
                compression_ratio=compression_ratio,
                num_levels=3,
                decay_factor=0.8
            )
        elif method.lower() == "fastv":
            num_centroids = max(int(self.hidden_size * compression_ratio), 8)
            return FastVCompressor(
                hidden_size=self.hidden_size,
                num_centroids=num_centroids,
                sparsity_threshold=0.2
            )
        elif method.lower() == "fastvideo":
            return FastVideoCompressor(
                hidden_size=self.hidden_size,
                keyframe_interval=8,
                compression_ratio=compression_ratio
            )
        elif method.lower() == "minillm":
            student_size = max(int(self.hidden_size * compression_ratio), 8)
            return MiniLLMCompressor(
                hidden_size=self.hidden_size,
                student_size=student_size,
                num_layers=2
            )
        else:
            print(f"Warning: Unknown compression method '{method}', using PyramidCompressor as default.")
            return PyramidCompressor(
                hidden_size=self.hidden_size,
                compression_ratio=compression_ratio
            )
    
    def _process_with_pikv(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process tensor through Distributed PiKV MoE."""
        try:
            # Get the shape of the input tensor
            if len(tensor.shape) == 4:  # [batch_size, num_heads, seq_len, head_dim]
                batch_size, num_heads, seq_len, head_dim = tensor.shape
                # Reshape to [batch_size, seq_len, hidden_size]
                hidden_size = num_heads * head_dim
                tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
            elif len(tensor.shape) == 3:  # [batch_size, seq_len, hidden_size]
                pass  # Already in correct shape
            elif len(tensor.shape) == 2:  # [batch_size, hidden_size]
                # Add sequence length dimension
                tensor = tensor.unsqueeze(1)  # [batch_size, 1, hidden_size]
            elif len(tensor.shape) == 1:  # [hidden_size]
                # Add batch and sequence length dimensions
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            
            # Ensure tensor has the correct hidden size
            if tensor.shape[-1] != self.hidden_size:
                # Project to hidden size if needed
                projection = nn.Linear(tensor.shape[-1], self.hidden_size).to(tensor.device)
                tensor = projection(tensor)
            
            # Process through Distributed PiKV MoE
            processed, _ = self.pikv(tensor, tensor)  # Pass tensor as both input and query
            
            # Reshape back to original dimensions if needed
            if len(tensor.shape) == 4:
                processed = processed.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
            
            return processed
        except Exception as e:
            print(f"Rank {self.rank}: Error in _process_with_pikv: {e}")
            raise
    
    def _compress_kv_cache(
        self, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """Compress KV cache using the selected compressor and measure performance."""
        # Record original size
        original_size = key.element_size() * key.nelement() + value.element_size() * value.nelement()
        
        # Measure compression time
        start_time = time.time()
        compressed_key, compressed_value = self.compressor(key, value, importance)
        compression_time = time.time() - start_time
        
        # Record compressed size
        compressed_size = compressed_key.element_size() * compressed_key.nelement() + compressed_value.element_size() * compressed_value.nelement()
        
        # Store metrics
        self.compression_times.append(compression_time)
        self.original_sizes.append(original_size)
        self.compressed_sizes.append(compressed_size)
        
        return compressed_key, compressed_value, compression_time, compressed_size / original_size
    
    def _compute_importance(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute importance scores for tokens in the sequence."""
        # Simple attention-based importance calculation
        # Higher norms typically correlate with more important tokens
        if len(tensor.shape) == 4:  # [batch_size, num_heads, seq_len, head_dim]
            # Average across heads and compute L2 norm
            importance = torch.norm(tensor.mean(dim=1), dim=-1)  # [batch_size, seq_len]
        elif len(tensor.shape) == 3:  # [batch_size, seq_len, hidden_size]
            importance = torch.norm(tensor, dim=-1)  # [batch_size, seq_len]
        else:
            # Default uniform importance if shape is unexpected
            importance = torch.ones(tensor.shape[0], 1, device=tensor.device)
        
        # Normalize to [0, 1]
        importance = importance / (importance.max() + 1e-6)
        
        return importance
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        with_metrics: bool = True
    ) -> Tuple[str, Dict[str, float]]:
        """Generate text using distributed PiKV cache with compression."""
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            input_ids = input_ids.to(self.device)
            
            # Clear metrics if collecting
            if with_metrics:
                self.compression_times = []
                self.decompression_times = []
                self.original_sizes = []
                self.compressed_sizes = []
                self.prediction_losses = []
            
            # Initialize output
            output_ids = input_ids.clone()
            self.current_length = input_ids.size(1)
            
            # Generate tokens
            for i in range(max_new_tokens):
                if self.rank == 0 and i % 10 == 0:
                    print(f"Generating token {i+1}/{max_new_tokens}")
                
                # Get model outputs
                outputs = self.model(
                    input_ids=output_ids,
                    use_cache=True,
                    return_dict=True
                )
                
                # Calculate loss for original outputs if collecting metrics
                if with_metrics:
                    with torch.no_grad():
                        # Get next token logits
                        next_token_logits = outputs.logits[:, -1, :]
                        # Get target token
                        if i < max_new_tokens - 1:
                            target_token = output_ids[:, -1]
                            # Calculate cross entropy loss
                            loss = torch.nn.functional.cross_entropy(
                                next_token_logits, 
                                target_token
                            ).item()
                            self.prediction_losses.append(loss)
                
                # Process each layer's KV cache
                new_past_key_values = []
                for layer_idx, layer_output in enumerate(outputs.past_key_values):
                    key, value = layer_output
                    
                    # Compute importance scores
                    importance = self._compute_importance(key)
                    
                    # Compress KV cache
                    compressed_key, compressed_value, _, _ = self._compress_kv_cache(key, value, importance)
                    
                    # Process through Distributed PiKV MoE
                    processed_key = self._process_with_pikv(compressed_key)
                    processed_value = self._process_with_pikv(compressed_value)
                    
                    # Create new tuple for this layer
                    new_past_key_values.append((processed_key, processed_value))
                
                # Update model's KV cache
                outputs.past_key_values = tuple(new_past_key_values)
                
                # Get next token logits
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to output
                output_ids = torch.cat([output_ids, next_token], dim=1)
                self.current_length += 1
                
                # Stop if we reach max length
                if self.current_length >= self.max_length:
                    break
            
            # Compute metrics if requested
            metrics = {}
            if with_metrics and len(self.compression_times) > 0:
                metrics = {
                    "avg_compression_time": np.mean(self.compression_times),
                    "avg_compression_ratio": np.mean([cs / os for cs, os in zip(self.compressed_sizes, self.original_sizes)]),
                    "memory_reduction_percent": (1 - np.mean([cs / os for cs, os in zip(self.compressed_sizes, self.original_sizes)])) * 100,
                    "avg_perplexity": math.exp(np.mean(self.prediction_losses)) if self.prediction_losses else 0
                }
                
                # Compute acceleration rate (based on memory reduction)
                # This is an approximation - actual acceleration depends on hardware
                metrics["estimated_acceleration"] = 1.0 / (0.3 + 0.7 * metrics["avg_compression_ratio"])
            
            # Decode and return generated text along with metrics
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True), metrics
        except Exception as e:
            print(f"Rank {self.rank}: Error in generate: {e}")
            raise
    
    def evaluate_compressors(
        self,
        prompt: str,
        compressors_to_test: List[str] = ["lora", "pyramid", "fastv", "fastvideo", "minillm"],
        compression_ratios: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
        max_tokens_per_test: int = 20,
        temperature: float = 0.7
    ) -> Dict[str, List[Dict[str, float]]]:
        """Evaluate multiple compressors with different compression ratios."""
        results = {}
        
        # Loop through compression methods
        for method in compressors_to_test:
            method_results = []
            
            # Loop through compression ratios
            for ratio in compression_ratios:
                print(f"Rank {self.rank}: Testing {method} with ratio {ratio}")
                
                # Create new compressor
                self.compressor = self._create_compressor(method, ratio)
                self.compressor = self.compressor.to(self.device)
                self.compression_method = method
                self.compression_ratio = ratio
                
                # Generate text and collect metrics
                try:
                    _, metrics = self.generate(
                        prompt,
                        max_new_tokens=max_tokens_per_test,
                        temperature=temperature,
                        with_metrics=True
                    )
                    
                    # Add compression method and ratio to metrics
                    metrics["method"] = method
                    metrics["compression_ratio"] = ratio
                    
                    # Add to results for this method
                    method_results.append(metrics)
                    
                    # Print key metrics
                    print(f"  - Compression ratio: {metrics['avg_compression_ratio']:.4f}")
                    print(f"  - Estimated acceleration: {metrics['estimated_acceleration']:.2f}x")
                    print(f"  - Perplexity: {metrics['avg_perplexity']:.4f}")
                except Exception as e:
                    print(f"Rank {self.rank}: Error evaluating {method} with ratio {ratio}: {e}")
                    # Add empty metrics to keep structure
                    method_results.append({
                        "method": method,
                        "compression_ratio": ratio,
                        "error": str(e)
                    })
            
            # Store results for this method
            results[method] = method_results
        
        # Save results to file
        if self.rank == 0:
            self._save_and_plot_results(results)
        
        return results
    
    def _save_and_plot_results(self, results: Dict[str, List[Dict[str, float]]]):
        """Save and plot evaluation results."""
        # Convert to DataFrame
        all_results = []
        for method, method_results in results.items():
            all_results.extend(method_results)
        
        df = pd.DataFrame(all_results)
        
        # Save to CSV
        df.to_csv("results/compression_evaluation.csv", index=False)
        print("Results saved to results/compression_evaluation.csv")
        
        # Create plots
        
        # 1. Compression Ratio vs. Perplexity
        plt.figure(figsize=(10, 6))
        for method in df["method"].unique():
            method_df = df[df["method"] == method]
            plt.plot(
                method_df["compression_ratio"], 
                method_df["avg_perplexity"],
                'o-',
                label=method
            )
        
        plt.xlabel("Compression Ratio")
        plt.ylabel("Perplexity (lower is better)")
        plt.title("Compression Ratio vs. Perplexity")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("results/compression_vs_perplexity.png")
        plt.close()
        
        # 2. Compression Ratio vs. Acceleration
        plt.figure(figsize=(10, 6))
        for method in df["method"].unique():
            method_df = df[df["method"] == method]
            plt.plot(
                method_df["compression_ratio"], 
                method_df["estimated_acceleration"],
                'o-',
                label=method
            )
        
        plt.xlabel("Compression Ratio")
        plt.ylabel("Estimated Acceleration")
        plt.title("Compression Ratio vs. Acceleration")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("results/compression_vs_acceleration.png")
        plt.close()
        
        # 3. Tradeoff plot (bubble chart)
        plt.figure(figsize=(12, 8))
        for method in df["method"].unique():
            method_df = df[df["method"] == method]
            
            # Use compression time as bubble size
            sizes = method_df["avg_compression_time"] * 1000 + 20  # Scale for visibility
            
            plt.scatter(
                method_df["compression_ratio"],
                method_df["avg_perplexity"],
                s=sizes,
                alpha=0.7,
                label=method
            )
            
            # Add method labels to points
            for i, row in method_df.iterrows():
                plt.annotate(
                    f"{row['method']}-{row['compression_ratio']:.1f}",
                    (row["compression_ratio"], row["avg_perplexity"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8
                )
        
        plt.xlabel("Compression Ratio (lower is better)")
        plt.ylabel("Perplexity (lower is better)")
        plt.title("Compression Methods Tradeoff: Ratio vs. Perplexity vs. Time")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig("results/compression_tradeoff.png")
        plt.close()
        
        print("Plots saved to results/ directory")

def main():
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Distributed PiKV Cache with Compression")
        parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
        parser.add_argument("--method", type=str, default="pyramid", choices=["lora", "lora+", "pyramid", "fastv", "fastvideo", "minillm"], help="Compression method")
        parser.add_argument("--ratio", type=float, default=0.5, help="Compression ratio")
        parser.add_argument("--evaluate", action="store_true", help="Run evaluation of multiple compressors")
        args = parser.parse_args()
        
        # Initialize distributed PiKV cache with compression
        print(f"Initializing DistributedPiKVCacheWithCompression using {args.method} compression...")
        pikv_cache = DistributedPiKVCacheWithCompression(
            model_name=args.model,
            max_length=1024,
            compression_method=args.method,
            compression_ratio=args.ratio
        )
        
        if args.evaluate:
            # Run evaluation if requested
            if pikv_cache.rank == 0:
                print("\nRunning compressor evaluation...")
            
            # Example prompt for evaluation
            prompt = "The future of artificial intelligence is"
            
            # Test different compressors
            results = pikv_cache.evaluate_compressors(
                prompt=prompt,
                compressors_to_test=["lora", "pyramid", "fastv", "fastvideo", "minillm"],
                compression_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
                max_tokens_per_test=20
            )
            
            if pikv_cache.rank == 0:
                print("\nEvaluation complete. Check results/ directory for detailed analysis.")
        else:
            # Example prompts for text generation
            prompts = [
                "The future of artificial intelligence is",
                "The relationship between compression ratio and model accuracy is",
                "When optimizing large language models, one should consider"
            ]
            
            # Generate text for each prompt
            for prompt in prompts:
                if pikv_cache.rank == 0:  # Only print on main process
                    print(f"\nPrompt: {prompt}")
                
                generated_text, metrics = pikv_cache.generate(
                    prompt,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    with_metrics=True
                )
                
                if pikv_cache.rank == 0:  # Only print on main process
                    print(f"Generated: {generated_text}")
                    print("\nCompression Metrics:")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
        
        # Clean up distributed environment
        dist.destroy_process_group()
        print("Distributed environment cleaned up")
    except Exception as e:
        print(f"Error in main: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise

if __name__ == "__main__":
    main() 