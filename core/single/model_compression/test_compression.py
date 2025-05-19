import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import compression modules
from core.single.model_compression.matrix_defactorization import LoRACompressor, LoRAPlusCompressor
from core.single.model_compression.cache_reduction import PyramidCompressor, FastVCompressor
from core.single.model_compression.distillation import FastVideoCompressor, MiniLLMCompressor
from core.single.model_compression.compression_utils import CompressionEvaluator, SimpleNextTokenPredictor
from core.single.config import config

def test_individual_compressors():
    """Test each compressor individually"""
    print("\n===== Testing Individual Compressors =====")
    
    # Initialize evaluator
    evaluator = CompressionEvaluator(hidden_size=config["hidden_size"])
    
    # Test data parameters
    batch_sizes = [8]
    seq_lens = [128]
    importance_levels = [0.2, 0.5, 0.8]
    
    # Test each compressor
    
    # 1. LoRA Compressor
    print("\nTesting LoRA Compressor:")
    lora_compressor = LoRACompressor(
        hidden_size=config["hidden_size"],
        rank=16,
        alpha=32.0,
        dropout=0.1
    ).to(evaluator.device)
    
    evaluator.benchmark_compressor(
        compressor_name="LoRA",
        compressor=lora_compressor,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        importance_levels=importance_levels
    )
    
    # 2. LoRA+ Compressor
    print("\nTesting LoRA+ Compressor:")
    loraplus_compressor = LoRAPlusCompressor(
        hidden_size=config["hidden_size"],
        ranks=[4, 8, 16],
        alpha=32.0,
        dropout=0.1
    ).to(evaluator.device)
    
    evaluator.benchmark_compressor(
        compressor_name="LoRA+",
        compressor=loraplus_compressor,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        importance_levels=importance_levels
    )
    
    # 3. Pyramid Compressor
    print("\nTesting Pyramid Compressor:")
    pyramid_compressor = PyramidCompressor(
        hidden_size=config["hidden_size"],
        compression_ratio=0.5,
        num_levels=3,
        decay_factor=0.8
    ).to(evaluator.device)
    
    evaluator.benchmark_compressor(
        compressor_name="Pyramid",
        compressor=pyramid_compressor,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        importance_levels=importance_levels
    )
    
    # 4. FastV Compressor
    print("\nTesting FastV Compressor:")
    fastv_compressor = FastVCompressor(
        hidden_size=config["hidden_size"],
        num_centroids=32,
        sparsity_threshold=0.2
    ).to(evaluator.device)
    
    evaluator.benchmark_compressor(
        compressor_name="FastV",
        compressor=fastv_compressor,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        importance_levels=importance_levels
    )
    
    # 5. FastVideo Compressor
    print("\nTesting FastVideo Compressor:")
    fastvideo_compressor = FastVideoCompressor(
        hidden_size=config["hidden_size"],
        keyframe_interval=8,
        motion_threshold=0.2,
        compression_ratio=0.5
    ).to(evaluator.device)
    
    evaluator.benchmark_compressor(
        compressor_name="FastVideo",
        compressor=fastvideo_compressor,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        importance_levels=importance_levels
    )
    
    # 6. MiniLLM Compressor
    print("\nTesting MiniLLM Compressor:")
    minillm_compressor = MiniLLMCompressor(
        hidden_size=config["hidden_size"],
        student_size=64,
        num_layers=2,
        use_attention=True
    ).to(evaluator.device)
    
    evaluator.benchmark_compressor(
        compressor_name="MiniLLM",
        compressor=minillm_compressor,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        importance_levels=importance_levels
    )
    
    # Save all results
    evaluator.save_results("individual_compressor_results.csv")
    
    print("\n===== Individual Compressor Testing Complete =====")

def compare_all_compressors():
    """Compare all compressors against each other"""
    print("\n===== Comparing All Compressors =====")
    
    # Initialize evaluator
    evaluator = CompressionEvaluator(hidden_size=config["hidden_size"])
    
    # Initialize all compressors
    compressors = {
        "LoRA": LoRACompressor(
            hidden_size=config["hidden_size"],
            rank=16,
            alpha=32.0
        ).to(evaluator.device),
        
        "LoRA+": LoRAPlusCompressor(
            hidden_size=config["hidden_size"],
            ranks=[4, 8, 16],
            alpha=32.0
        ).to(evaluator.device),
        
        "Pyramid": PyramidCompressor(
            hidden_size=config["hidden_size"],
            compression_ratio=0.5,
            num_levels=3
        ).to(evaluator.device),
        
        "FastV": FastVCompressor(
            hidden_size=config["hidden_size"],
            num_centroids=32
        ).to(evaluator.device),
        
        "FastVideo": FastVideoCompressor(
            hidden_size=config["hidden_size"],
            keyframe_interval=8,
            compression_ratio=0.5
        ).to(evaluator.device),
        
        "MiniLLM": MiniLLMCompressor(
            hidden_size=config["hidden_size"],
            student_size=64,
            num_layers=2
        ).to(evaluator.device)
    }
    
    # Run comparison
    comparison_results = evaluator.compare_compressors(
        compressors=compressors,
        batch_size=8,
        seq_len=128,
        importance_level=0.5,
        repeat=10
    )
    
    # Save results
    evaluator.save_results("compressor_comparison_results.csv")
    
    print("\n===== Compressor Comparison Complete =====")
    
    return comparison_results

def run_next_token_prediction_test():
    """Test compressors on next token prediction task"""
    print("\n===== Running Next Token Prediction Test =====")
    
    # Initialize evaluator
    evaluator = CompressionEvaluator(hidden_size=config["hidden_size"])
    
    # Initialize all compressors
    compressors = {
        "LoRA": LoRACompressor(
            hidden_size=config["hidden_size"],
            rank=16,
            alpha=32.0
        ).to(evaluator.device),
        
        "Pyramid": PyramidCompressor(
            hidden_size=config["hidden_size"],
            compression_ratio=0.5,
            num_levels=3
        ).to(evaluator.device),
        
        "FastV": FastVCompressor(
            hidden_size=config["hidden_size"],
            num_centroids=32
        ).to(evaluator.device),
        
        "MiniLLM": MiniLLMCompressor(
            hidden_size=config["hidden_size"],
            student_size=64,
            num_layers=2
        ).to(evaluator.device)
    }
    
    # Initialize predictor
    predictor = SimpleNextTokenPredictor(
        hidden_size=config["hidden_size"],
        vocab_size=config["vocab_size"]
    ).to(evaluator.device)
    
    # Run test
    prediction_results = evaluator.run_next_token_prediction_test(
        compressors=compressors,
        predictor=predictor,
        vocab_size=config["vocab_size"],
        seq_len=64,
        batch_size=4
    )
    
    # Save results
    evaluator.save_results("next_token_prediction_results.csv")
    
    print("\n===== Next Token Prediction Test Complete =====")
    
    return prediction_results

def analyze_accuracy_compression_tradeoff(prediction_results):
    """Analyze the tradeoff between compression and accuracy"""
    print("\n===== Analyzing Accuracy/Compression Tradeoff =====")
    
    # Extract data
    names = list(prediction_results.keys())
    compression_ratios = [prediction_results[name]['compression_ratio'] for name in names]
    accuracies = [prediction_results[name]['accuracy'] for name in names]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(compression_ratios, accuracies, s=100, alpha=0.7)
    
    # Add labels
    for i, name in enumerate(names):
        plt.annotate(
            name,
            (compression_ratios[i], accuracies[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=12
        )
    
    # Add trend line
    z = np.polyfit(compression_ratios, accuracies, 1)
    p = np.poly1d(z)
    plt.plot(compression_ratios, p(compression_ratios), "r--", alpha=0.7)
    
    # Add labels and title
    plt.title('Compression vs. Accuracy Tradeoff', fontsize=14)
    plt.xlabel('Compression Ratio (smaller is better)', fontsize=12)
    plt.ylabel('Prediction Accuracy', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Save plot
    plt.savefig("results/compression_accuracy_tradeoff.png")
    plt.close()
    
    print("Tradeoff analysis complete. Plot saved to results/compression_accuracy_tradeoff.png")
    
    # Calculate correlation
    correlation = np.corrcoef(compression_ratios, accuracies)[0, 1]
    print(f"Correlation between compression ratio and accuracy: {correlation:.4f}")
    
    print("\n===== Tradeoff Analysis Complete =====")

def main():
    """Main function to run all tests"""
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Print PyTorch and CUDA information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test individual compressors
    test_individual_compressors()
    
    # Compare all compressors
    comparison_results = compare_all_compressors()
    
    # Run next token prediction test
    prediction_results = run_next_token_prediction_test()
    
    # Analyze accuracy/compression tradeoff
    analyze_accuracy_compression_tradeoff(prediction_results)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 