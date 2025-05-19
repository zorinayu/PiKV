import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd

class CompressionEvaluator:
    """
    Evaluation utilities for KV cache compression methods
    """
    def __init__(self, hidden_size: int = 256, device: Optional[torch.device] = None):
        self.hidden_size = hidden_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = []
        
        # Create output directory
        os.makedirs("results", exist_ok=True)
    
    def generate_test_data(
        self,
        batch_size: int = 8,
        seq_len: int = 128,
        importance_level: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate test data for compression evaluation"""
        keys = torch.randn(batch_size, seq_len, self.hidden_size, device=self.device)
        values = torch.randn(batch_size, seq_len, self.hidden_size, device=self.device)
        
        # Generate importance scores
        if importance_level < 0:  # Random importance
            importance = torch.rand(batch_size, seq_len, device=self.device)
        else:  # Fixed importance level
            importance = torch.ones(batch_size, seq_len, device=self.device) * importance_level
        
        return keys, values, importance
    
    def measure_compression_performance(
        self,
        compressor: nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
        repeat: int = 10
    ) -> Dict[str, Any]:
        """Measure compression performance and quality"""
        # Prepare results dictionary
        results = {}
        
        # Cache original data for accuracy loss computation
        keys_cpu = keys.detach().cpu()
        values_cpu = values.detach().cpu()
        
        # Warmup runs
        for _ in range(3):
            with torch.no_grad():
                _ = compressor(keys, values, importance)
        
        # Measure compression time
        start_time = time.time()
        for _ in range(repeat):
            with torch.no_grad():
                compressed_keys, compressed_values = compressor(keys, values, importance)
        compression_time = (time.time() - start_time) / repeat
        
        # Calculate compression ratio and memory usage
        original_size = keys.element_size() * keys.nelement() + values.element_size() * values.nelement()
        compressed_size = compressed_keys.element_size() * compressed_keys.nelement() + compressed_values.element_size() * compressed_values.nelement()
        compression_ratio = compressed_size / original_size
        memory_saved = 1.0 - compression_ratio
        
        # Calculate compression quality (MSE and similarity)
        with torch.no_grad():
            # MSE loss
            key_mse = F.mse_loss(compressed_keys.cpu(), keys_cpu).item()
            value_mse = F.mse_loss(compressed_values.cpu(), values_cpu).item()
            avg_mse = (key_mse + value_mse) / 2
            
            # Cosine similarity
            keys_flat = keys_cpu.reshape(-1, self.hidden_size)
            compressed_keys_flat = compressed_keys.cpu().reshape(-1, self.hidden_size)
            values_flat = values_cpu.reshape(-1, self.hidden_size)
            compressed_values_flat = compressed_values.cpu().reshape(-1, self.hidden_size)
            
            key_sim = F.cosine_similarity(keys_flat, compressed_keys_flat).mean().item()
            value_sim = F.cosine_similarity(values_flat, compressed_values_flat).mean().item()
            avg_sim = (key_sim + value_sim) / 2
        
        # Collect results
        results["compression_time"] = compression_time
        results["compression_ratio"] = compression_ratio
        results["memory_saved"] = memory_saved
        results["key_mse"] = key_mse
        results["value_mse"] = value_mse
        results["avg_mse"] = avg_mse
        results["key_sim"] = key_sim
        results["value_sim"] = value_sim
        results["avg_sim"] = avg_sim
        
        # Add compressor-specific stats if available
        if hasattr(compressor, "get_compression_stats"):
            compressor_stats = compressor.get_compression_stats()
            for key, value in compressor_stats.items():
                if isinstance(value, (int, float)):
                    results[f"compressor_{key}"] = value
        
        return results
    
    def benchmark_compressor(
        self,
        compressor_name: str,
        compressor: nn.Module,
        batch_sizes: List[int] = [8],
        seq_lens: List[int] = [128],
        importance_levels: List[float] = [0.5],
        repeat: int = 10
    ) -> List[Dict[str, Any]]:
        """Benchmark a compressor with various settings"""
        test_results = []
        
        # Run benchmark for all combinations
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for importance_level in importance_levels:
                    # Generate test data
                    keys, values, importance = self.generate_test_data(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        importance_level=importance_level
                    )
                    
                    # Measure performance
                    result = self.measure_compression_performance(
                        compressor=compressor,
                        keys=keys,
                        values=values,
                        importance=importance,
                        repeat=repeat
                    )
                    
                    # Add test parameters
                    result["compressor"] = compressor_name
                    result["batch_size"] = batch_size
                    result["seq_len"] = seq_len
                    result["importance_level"] = importance_level
                    result["hidden_size"] = self.hidden_size
                    
                    # Print single test result
                    print(f"\nTest {len(test_results)+1}:")
                    print(f"Compressor: {compressor_name}, Batch: {batch_size}, Seq: {seq_len}, Imp: {importance_level:.2f}")
                    print(f"Compression Ratio: {result['compression_ratio']:.4f} (Memory Saved: {result['memory_saved']:.2%})")
                    print(f"Avg MSE: {result['avg_mse']:.6f}, Avg Similarity: {result['avg_sim']:.4f}")
                    print(f"Compression Time: {result['compression_time']*1000:.2f} ms")
                    
                    # Save results
                    test_results.append(result)
                    self.results.append(result)
        
        return test_results
    
    def compare_compressors(
        self,
        compressors: Dict[str, nn.Module],
        batch_size: int = 8,
        seq_len: int = 128,
        importance_level: float = 0.5,
        repeat: int = 10
    ):
        """Compare multiple compressors on the same data"""
        print("\n===== Compressor Comparison =====")
        
        # Generate test data (same for all compressors)
        keys, values, importance = self.generate_test_data(
            batch_size=batch_size,
            seq_len=seq_len,
            importance_level=importance_level
        )
        
        # Test metrics to compare
        comparison_metrics = ['compression_ratio', 'memory_saved', 'avg_mse', 'avg_sim', 'compression_time']
        
        # Initialize results
        comparison_results = {name: {} for name in compressors.keys()}
        
        # Run tests for each compressor
        for name, compressor in compressors.items():
            print(f"\nEvaluating: {name}")
            
            # Measure performance
            result = self.measure_compression_performance(
                compressor=compressor,
                keys=keys,
                values=values,
                importance=importance,
                repeat=repeat
            )
            
            # Extract relevant metrics
            for metric in comparison_metrics:
                comparison_results[name][metric] = result[metric]
                print(f"  {metric}: {result[metric]:.4f}")
            
            # Save full results
            result["compressor"] = name
            result["batch_size"] = batch_size
            result["seq_len"] = seq_len
            result["importance_level"] = importance_level
            result["hidden_size"] = self.hidden_size
            self.results.append(result)
        
        # Plot comparison
        self.plot_compression_comparison(comparison_results, save_path="results/compressor_comparison.png")
        
        print("\n===== Comparison Complete =====")
        return comparison_results
    
    def run_next_token_prediction_test(
        self,
        compressors: Dict[str, nn.Module],
        predictor: nn.Module,
        vocab_size: int = 50257,
        seq_len: int = 128,
        batch_size: int = 4,
        num_rounds: int = 5
    ):
        """Test compressors on next token prediction task"""
        print("\n===== Next Token Prediction Test =====")
        
        # Create test data for sequence modeling
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        
        # Create embeddings from input IDs (simulate a model's embedding layer)
        embedding_dim = self.hidden_size
        embedding = nn.Embedding(vocab_size, embedding_dim).to(self.device)
        embeddings = embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Create keys and values from embeddings
        keys = F.linear(embeddings, torch.randn(embedding_dim, embedding_dim, device=self.device))
        values = F.linear(embeddings, torch.randn(embedding_dim, embedding_dim, device=self.device))
        
        # Compute uncompressed baseline first
        baseline_loss, baseline_accuracy = self._evaluate_next_token_prediction(
            predictor, keys, values, targets
        )
        
        # Results dictionary
        prediction_results = {
            "baseline": {"loss": baseline_loss, "accuracy": baseline_accuracy, "compression_ratio": 1.0}
        }
        
        # Test each compressor
        for name, compressor in compressors.items():
            print(f"\nTesting {name} on next token prediction...")
            
            # Apply compression
            with torch.no_grad():
                compressed_keys, compressed_values = compressor(keys, values)
            
            # Evaluate prediction with compressed KV cache
            loss, accuracy = self._evaluate_next_token_prediction(
                predictor, compressed_keys, compressed_values, targets
            )
            
            # Calculate compression metrics
            original_size = keys.nelement() * keys.element_size() + values.nelement() * values.element_size()
            compressed_size = compressed_keys.nelement() * compressed_keys.element_size() + compressed_values.nelement() * compressed_values.element_size()
            compression_ratio = compressed_size / original_size
            
            # Store results
            prediction_results[name] = {
                "loss": loss,
                "accuracy": accuracy,
                "compression_ratio": compression_ratio,
                "loss_increase": (loss - baseline_loss) / baseline_loss,
                "accuracy_decrease": (baseline_accuracy - accuracy)
            }
            
            # Print results
            print(f"  Loss: {loss:.4f} (vs. baseline {baseline_loss:.4f})")
            print(f"  Accuracy: {accuracy:.4f} (vs. baseline {baseline_accuracy:.4f})")
            print(f"  Compression Ratio: {compression_ratio:.4f}")
        
        # Plot results
        self.plot_prediction_results(prediction_results, save_path="results/prediction_results.png")
        
        print("\n===== Next Token Prediction Test Complete =====")
        return prediction_results
    
    def _evaluate_next_token_prediction(
        self,
        predictor: nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[float, float]:
        """Evaluate next token prediction using the given keys and values"""
        with torch.no_grad():
            # Get predictions
            logits = predictor(keys, values)  # Should return [batch_size, seq_len, vocab_size]
            
            # Calculate loss
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets).float().mean()
            
            return loss.item(), correct.item()
    
    def plot_compression_comparison(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """Plot comparison of different compressors"""
        # Metrics to plot
        metrics = ['compression_ratio', 'avg_mse', 'avg_sim', 'compression_time']
        
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            
            # Extract values for this metric
            names = list(comparison_results.keys())
            values = [comparison_results[name][metric] for name in names]
            
            # Convert time to milliseconds for better readability
            if metric == 'compression_time':
                values = [v * 1000 for v in values]  # Convert to ms
                ylabel = 'Time (ms)'
            elif metric == 'compression_ratio':
                ylabel = 'Ratio (smaller is better)'
            elif metric == 'avg_mse':
                ylabel = 'MSE (smaller is better)'
            elif metric == 'avg_sim':
                ylabel = 'Similarity (larger is better)'
            else:
                ylabel = metric
            
            # Create bar chart
            bars = plt.bar(names, values, alpha=0.7)
            
            # Add value labels
            for bar, value in zip(bars, values):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{value:.4f}',
                    ha='center',
                    va='bottom',
                    rotation=45 if metric == 'compression_time' else 0
                )
            
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.ylabel(ylabel)
            plt.xticks(rotation=45, ha='right')
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Comparison plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_prediction_results(
        self,
        prediction_results: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """Plot next token prediction results"""
        plt.figure(figsize=(15, 6))
        
        # Extract data
        names = list(prediction_results.keys())
        compression_ratios = [prediction_results[name]['compression_ratio'] for name in names]
        accuracies = [prediction_results[name]['accuracy'] for name in names]
        
        # Create subplots
        plt.subplot(1, 2, 1)
        
        # Plot accuracy vs. compression ratio
        plt.scatter(compression_ratios, accuracies, s=100, alpha=0.7)
        
        # Add labels for each point
        for i, name in enumerate(names):
            plt.annotate(
                name,
                (compression_ratios[i], accuracies[i]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.title('Accuracy vs. Compression Ratio')
        plt.xlabel('Compression Ratio (smaller is better)')
        plt.ylabel('Prediction Accuracy')
        plt.grid(alpha=0.3)
        
        # Second subplot - bar chart of accuracies
        plt.subplot(1, 2, 2)
        
        bars = plt.bar(names, accuracies, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, accuracies):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f'{value:.4f}',
                ha='center',
                va='bottom'
            )
        
        plt.title('Prediction Accuracy by Compressor')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Prediction results plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def save_results(self, filename: str = "compression_results.csv"):
        """Save benchmark results to CSV"""
        if not self.results:
            print("No results to save.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save to CSV
        csv_path = os.path.join("results", filename)
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        return csv_path

class SimpleNextTokenPredictor(nn.Module):
    """
    Simple next token prediction model for testing compression
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super(SimpleNextTokenPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Simple attention layer
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, vocab_size)
        )
    
    def forward(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Predict next tokens using the given keys and values
        
        Args:
            keys: Key tensors [batch_size, seq_len, hidden_size]
            values: Value tensors [batch_size, seq_len, hidden_size]
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, hidden_size = keys.shape
        
        # Generate queries from values
        queries = self.query_proj(values)
        
        # Apply attention
        attn_output, _ = self.attention(queries, keys, values)
        
        # Project to vocabulary
        logits = self.output_proj(attn_output)
        
        return logits 