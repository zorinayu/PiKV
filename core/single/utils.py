import torch
from config import config
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
import psutil
import os

def generate_data(batch_size, input_size):
    """Generate random data for training."""
    return torch.randn(batch_size, input_size)

def train_model(model, data, target, retain_graph=False):
    """Training loop."""
    # Clone the data to create a new computation graph
    data_clone = data.clone().detach()
    target_clone = target.clone().detach()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    optimizer.zero_grad()
    output = model(data_clone)
    loss = criterion(output, target_clone)
    loss.backward(retain_graph=retain_graph)
    optimizer.step()
    return loss.item()

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def visualize_kv_cache(model, save_path=None):
    """Visualize KV cache usage for PiKV model."""
    if not hasattr(model, 'kv_caches'):
        return None
    
    # Get cache sizes and usage
    cache_sizes = [cache.size for cache in model.kv_caches]
    cache_usage = []
    
    for cache in model.kv_caches:
        # Calculate usage based on non-zero values
        if hasattr(cache, 'values'):
            usage = (cache.values != 0).sum().item() / cache.size
            cache_usage.append(usage)
        else:
            cache_usage.append(0)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    # Plot cache sizes
    plt.subplot(1, 2, 1)
    plt.bar(range(len(cache_sizes)), cache_sizes, color='blue', alpha=0.7)
    plt.title('Pyramidal Cache Sizes')
    plt.xlabel('Layer')
    plt.ylabel('Cache Size')
    
    # Plot cache usage
    plt.subplot(1, 2, 2)
    plt.bar(range(len(cache_usage)), cache_usage, color='green', alpha=0.7)
    plt.title('Cache Usage')
    plt.xlabel('Layer')
    plt.ylabel('Usage Ratio')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return cache_sizes, cache_usage

def compare_performance(model_pikv, model_standard, data, target, num_runs=5):
    """Compare performance metrics between PiKV and Standard MoE models."""
    # Initialize data structures for raw measurements
    pikv_times = []
    pikv_memories = []
    pikv_losses = []
    
    standard_times = []
    standard_memories = []
    standard_losses = []
    
    # Measure PiKV performance
    for _ in range(num_runs):
        # Record initial memory
        initial_memory = get_memory_usage()
        
        # Time the forward pass
        start_time = time.time()
        loss_pikv = train_model(model_pikv, data, target, retain_graph=True)
        end_time = time.time()
        
        # Record final memory
        final_memory = get_memory_usage()
        
        # Store results
        pikv_times.append(end_time - start_time)
        pikv_memories.append(final_memory - initial_memory)
        pikv_losses.append(loss_pikv)
    
    # Measure Standard MoE performance
    for _ in range(num_runs):
        # Record initial memory
        initial_memory = get_memory_usage()
        
        # Time the forward pass
        start_time = time.time()
        loss_standard = train_model(model_standard, data, target, retain_graph=False)
        end_time = time.time()
        
        # Record final memory
        final_memory = get_memory_usage()
        
        # Store results
        standard_times.append(end_time - start_time)
        standard_memories.append(final_memory - initial_memory)
        standard_losses.append(loss_standard)
    
    # Calculate averages and create results dictionary
    results = {
        'pikv': {
            'time': pikv_times,
            'memory': pikv_memories,
            'loss': pikv_losses,
            'avg_time': float(np.mean(pikv_times)),
            'avg_memory': float(np.mean(pikv_memories)),
            'avg_loss': float(np.mean(pikv_losses))
        },
        'standard': {
            'time': standard_times,
            'memory': standard_memories,
            'loss': standard_losses,
            'avg_time': float(np.mean(standard_times)),
            'avg_memory': float(np.mean(standard_memories)),
            'avg_loss': float(np.mean(standard_losses))
        }
    }
    
    return results

def plot_performance_comparison(results, save_path=None):
    """Plot performance comparison between models."""
    metrics = ['time', 'memory', 'loss']
    model_types = ['pikv', 'standard']
    
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        
        values = [results[model_type][f'avg_{metric}'] for model_type in model_types]
        plt.bar(model_types, values, color=['blue', 'green'], alpha=0.7)
        
        plt.title(f'Average {metric.capitalize()}')
        plt.ylabel(metric.capitalize())
        
        # Add value labels
        for j, v in enumerate(values):
            plt.text(j, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
