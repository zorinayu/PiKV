import torch
import os
from config import config
from pikv_moe import PiKVMoE
from normal_moe import StandardMoE
from utils import generate_data, train_model, visualize_kv_cache, compare_performance, plot_performance_comparison

def compare_models():
    # Create output directory for visualizations
    os.makedirs('output', exist_ok=True)
    
    # Generate some random data for training
    data = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    target = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    
    # Initialize models
    model_pikv = PiKVMoE().to(config['device'])
    model_standard = StandardMoE().to(config['device'])
    
    # Train both models and compare losses
    for epoch in range(config['epochs']):
        # Train PiKV model with retain_graph=True
        loss_pikv = train_model(model_pikv, data, target, retain_graph=True)
        
        # Train Standard MoE model (no need to retain graph for the last model)
        loss_standard = train_model(model_standard, data, target, retain_graph=False)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - PiKV Loss: {loss_pikv:.4f}, Standard MoE Loss: {loss_standard:.4f}")
    
    # Visualize KV cache usage for PiKV model
    print("Visualizing KV cache usage...")
    visualize_kv_cache(model_pikv, save_path='output/kv_cache_usage.png')
    
    # Compare performance metrics
    print("Comparing performance metrics...")
    results = compare_performance(model_pikv, model_standard, data, target, num_runs=5)
    
    # Plot performance comparison
    plot_performance_comparison(results, save_path='output/performance_comparison.png')
    
    # Print detailed results
    print("\nPerformance Comparison Results:")
    for model_type in ['pikv', 'standard']:
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
