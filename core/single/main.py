import torch
import os
from config import config
from pikv_moe import PiKVMoE
from normal_moe import StandardMoE
from lora import LoRAPiKVMoE
from utils import generate_data, train_model, visualize_kv_cache, compare_performance, plot_performance_comparison

def compare_models():
    # Create output directory for visualizations
    os.makedirs('output', exist_ok=True)
    
    # Generate separate random data for each model to avoid graph conflicts
    data_pikv = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    target_pikv = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    
    data_standard = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    target_standard = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    
    data_lora = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    target_lora = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    
    # Initialize models
    model_pikv = PiKVMoE().to(config['device'])
    model_standard = StandardMoE().to(config['device'])
    model_lora = LoRAPiKVMoE(rank=4, alpha=1.0).to(config['device'])
    
    # Train all models and compare losses
    for epoch in range(config['epochs']):
        # Train each model with its own data
        loss_pikv = train_model(model_pikv, data_pikv, target_pikv, retain_graph=True)
        loss_standard = train_model(model_standard, data_standard, target_standard, retain_graph=True)
        loss_lora = train_model(model_lora, data_lora, target_lora, retain_graph=True)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - PiKV Loss: {loss_pikv:.4f}, Standard MoE Loss: {loss_standard:.4f}, LoRA PiKV Loss: {loss_lora:.4f}")
    
    # Visualize KV cache usage for PiKV models
    print("Visualizing KV cache usage for standard PiKV...")
    visualize_kv_cache(model_pikv, save_path='output/kv_cache_usage_standard.png')
    
    print("Visualizing KV cache usage for LoRA PiKV...")
    visualize_kv_cache(model_lora, save_path='output/kv_cache_usage_lora.png')
    
    # Compare performance metrics
    print("Comparing performance metrics...")
    # Use a fresh data tensor for performance comparison
    data_compare = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    target_compare = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    
    results = compare_performance(model_pikv, model_standard, data_compare, target_compare, num_runs=5)
    
    # Add LoRA model to results
    lora_results = compare_performance(model_lora, model_standard, data_compare, target_compare, num_runs=5)
    results['lora'] = lora_results['pikv']
    
    # Plot performance comparison
    plot_performance_comparison(results, save_path='output/performance_comparison.png')
    
    # Print detailed results
    print("\nPerformance Comparison Results:")
    for model_type in ['pikv', 'standard', 'lora']:
        print(f"\n{model_type.upper()} Model:")
        for metric in ['time', 'memory', 'loss']:
            avg_value = results[model_type][f'avg_{metric}']
            if metric == 'time':
                print(f"  Average {metric}: {avg_value:.4f} seconds")
            elif metric == 'memory':
                print(f"  Average {metric}: {avg_value:.2f} MB")
            else:
                print(f"  Average {metric}: {avg_value:.4f}")

if __name__ == "__main__":
    compare_models()
