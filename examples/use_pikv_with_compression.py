#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use PiKV with model compression
"""

import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Dict, List

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required PiKV modules
from core.single.config import config
from core.single.pikv_moe import PiKVMoE
from core.single.model_compression import PyramidCompressor, LoRACompressor, MiniLLMCompressor
from core.single.model_compression import CompressionEvaluator, SimpleNextTokenPredictor

def example_next_token_prediction():
    """
    Simple example of using PiKV with compression for next token prediction
    """
    print("Running PiKV with compression for next token prediction...")
    
    # Initialize PiKV model
    pikv_model = PiKVMoE().to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize compressors
    compressors = {
        "None": None,  # Baseline (no compression)
        "LoRA": LoRACompressor(
            hidden_size=config["hidden_size"],
            rank=16,
            alpha=32.0
        ),
        "Pyramid": PyramidCompressor(
            hidden_size=config["hidden_size"],
            compression_ratio=0.5,
            num_levels=3
        ),
        "MiniLLM": MiniLLMCompressor(
            hidden_size=config["hidden_size"],
            student_size=64,
            num_layers=2
        )
    }
    
    # Move compressors to the correct device
    device = next(pikv_model.parameters()).device
    for name, compressor in compressors.items():
        if compressor is not None:
            compressors[name] = compressor.to(device)
    
    # Generate some random input data
    batch_size = 4
    seq_len = 32
    
    # Create random input tokens and target tokens
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
    
    # Create a simple embedding layer to convert token IDs to embeddings
    embedding = nn.Embedding(config["vocab_size"], config["hidden_size"]).to(device)
    
    # Convert input tokens to embeddings
    input_embeds = embedding(input_ids)  # [batch_size, seq_len, hidden_size]
    
    # Run PiKV to get KV cache
    with torch.no_grad():
        keys_values, _ = pikv_model(input_embeds)
    
    # Split into keys and values (assume equal split of the hidden dimension)
    hidden_size = config["hidden_size"]
    keys = keys_values[:, :, :hidden_size//2]
    values = keys_values[:, :, hidden_size//2:]
    
    # Initialize predictor for next token prediction
    predictor = SimpleNextTokenPredictor(hidden_size=config["hidden_size"]//2, vocab_size=config["vocab_size"]).to(device)
    
    # Results dictionary to store accuracy and time metrics
    results = {}
    
    # Test each compressor
    for name, compressor in compressors.items():
        print(f"\nTesting {name} compressor...")
        
        # Apply compression if compressor is not None
        if compressor is not None:
            start_time = time.time()
            with torch.no_grad():
                compressed_keys, compressed_values = compressor(keys, values)
            compression_time = time.time() - start_time
            
            # Calculate compression ratio
            original_size = keys.nelement() * keys.element_size() + values.nelement() * values.element_size()
            compressed_size = compressed_keys.nelement() * compressed_keys.element_size() + compressed_values.nelement() * compressed_values.element_size()
            compression_ratio = compressed_size / original_size
            
            print(f"Compression ratio: {compression_ratio:.4f} (saved {1-compression_ratio:.2%})")
            print(f"Compression time: {compression_time*1000:.2f} ms")
            
            # Use compressed KV cache for prediction
            prediction_keys = compressed_keys
            prediction_values = compressed_values
        else:
            # Use original KV cache for prediction
            prediction_keys = keys
            prediction_values = values
            compression_ratio = 1.0
            compression_time = 0.0
        
        # Predict next tokens
        with torch.no_grad():
            start_time = time.time()
            logits = predictor(prediction_keys, prediction_values)
            prediction_time = time.time() - start_time
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == target_ids).float().mean().item()
            
            print(f"Prediction time: {prediction_time*1000:.2f} ms")
            print(f"Accuracy: {accuracy:.4f}")
            
            # Store results
            results[name] = {
                "compression_ratio": compression_ratio,
                "compression_time": compression_time,
                "prediction_time": prediction_time,
                "accuracy": accuracy
            }
    
    # Plot results
    plot_results(results)

def plot_results(results: Dict[str, Dict[str, float]]):
    """Plot comparison results"""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get compressor names
    names = list(results.keys())
    
    # Plot compression ratio vs accuracy
    compression_ratios = [results[name]["compression_ratio"] for name in names]
    accuracies = [results[name]["accuracy"] for name in names]
    
    ax1.scatter(compression_ratios, accuracies, s=100)
    for i, name in enumerate(names):
        ax1.annotate(name, (compression_ratios[i], accuracies[i]), fontsize=10, 
                     xytext=(5, 5), textcoords="offset points")
    
    ax1.set_xlabel("Compression Ratio")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Compression Ratio vs Accuracy")
    ax1.grid(alpha=0.3)
    
    # Plot prediction time
    times = [results[name]["prediction_time"] * 1000 for name in names]  # Convert to ms
    
    bars = ax2.bar(names, times)
    ax2.set_xlabel("Compressor")
    ax2.set_ylabel("Prediction Time (ms)")
    ax2.set_title("Prediction Time by Compressor")
    ax2.grid(alpha=0.3)
    
    # Add time labels above bars
    for i, bar in enumerate(bars):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{times[i]:.2f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save plot
    plt.savefig("results/compression_comparison.png")
    print("\nResults plot saved to results/compression_comparison.png")
    
    # Show plot
    plt.show()

def main():
    """Main function"""
    # Print PyTorch and CUDA information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run example
    example_next_token_prediction()

if __name__ == "__main__":
    main() 