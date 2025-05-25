import torch
import os
from config import config
from pikv_moe import PiKVMoE
from normal_moe import StandardMoE
from lora import LoRAPiKVMoE
from distillation import create_teacher_model, distillation_training_step
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
    
    data_distill = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    target_distill = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    
    # Initialize models
    model_pikv = PiKVMoE().to(config['device'])
    model_standard = StandardMoE().to(config['device'])
    model_lora = LoRAPiKVMoE(rank=4, alpha=1.0).to(config['device'])
    
    # Initialize PiKV model with knowledge distillation
    model_distill = PiKVMoE(
        rank=4, 
        alpha=1.0, 
        use_distillation=True,
        teacher_hidden_size=config['hidden_size'] * 2
    ).to(config['device'])
    
    print("Models initialized:")
    print("- Standard PiKV MoE")
    print("- Standard MoE")
    print("- LoRA PiKV MoE")
    print("- PiKV MoE with Knowledge Distillation")
    
    # Train all models and compare losses
    for epoch in range(config['epochs']):
        # Train each model with its own data
        loss_pikv = train_model(model_pikv, data_pikv, target_pikv, retain_graph=True)
        loss_standard = train_model(model_standard, data_standard, target_standard, retain_graph=True)
        loss_lora = train_model(model_lora, data_lora, target_lora, retain_graph=True)
        
        # Train distillation model using the distillation step
        distill_loss_info = model_distill.distillation_step(
            input_data=data_distill,
            targets=None,  # No targets for this demo
            optimizer=torch.optim.Adam(model_distill.parameters(), lr=config['learning_rate'])
        )
        loss_distill = distill_loss_info.get('total_distill_loss', 0.0)
        
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"PiKV Loss: {loss_pikv:.4f}, "
              f"Standard MoE Loss: {loss_standard:.4f}, "
              f"LoRA PiKV Loss: {loss_lora:.4f}, "
              f"Distill PiKV Loss: {loss_distill:.4f}")
    
    # Visualize KV cache usage for PiKV models
    print("Visualizing KV cache usage for standard PiKV...")
    visualize_kv_cache(model_pikv, save_path='output/kv_cache_usage_standard.png')
    
    print("Visualizing KV cache usage for LoRA PiKV...")
    visualize_kv_cache(model_lora, save_path='output/kv_cache_usage_lora.png')
    
    print("Visualizing KV cache usage for Distillation PiKV...")
    visualize_kv_cache(model_distill, save_path='output/kv_cache_usage_distill.png')
    
    # Compare performance metrics
    print("Comparing performance metrics...")
    # Use a fresh data tensor for performance comparison
    data_compare = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    target_compare = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    
    results = compare_performance(model_pikv, model_standard, data_compare, target_compare, num_runs=5)
    
    # Add LoRA model to results
    lora_results = compare_performance(model_lora, model_standard, data_compare, target_compare, num_runs=5)
    results['lora'] = lora_results['pikv']
    
    # Add distillation model to results
    distill_results = compare_performance(model_distill, model_standard, data_compare, target_compare, num_runs=5)
    results['distillation'] = distill_results['pikv']
    
    # Plot performance comparison
    plot_performance_comparison(results, save_path='output/performance_comparison.png')
    
    # Print detailed results
    print("\nPerformance Comparison Results:")
    for model_type in ['pikv', 'standard', 'lora', 'distillation']:
        if model_type in results:
            print(f"\n{model_type.upper()} Model:")
            for metric in ['time', 'memory', 'loss']:
                avg_value = results[model_type][f'avg_{metric}']
                if metric == 'time':
                    print(f"  Average {metric}: {avg_value:.4f} seconds")
                elif metric == 'memory':
                    print(f"  Average {metric}: {avg_value:.2f} MB")
                else:
                    print(f"  Average {metric}: {avg_value:.4f}")

def demonstrate_distillation():
    """
    Demonstrate knowledge distillation functionality
    """
    print("\n" + "="*60)
    print("Knowledge Distillation Demonstration")
    print("="*60)
    
    # Create student model with distillation enabled
    student_model = PiKVMoE(
        rank=4,
        alpha=1.0,
        use_distillation=True,
        teacher_hidden_size=config['hidden_size'] * 2
    ).to(config['device'])
    
    # Generate training data
    train_data = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    
    # Demonstrate distillation training
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    
    print("Training with knowledge distillation...")
    for epoch in range(3):
        loss_info = student_model.distillation_step(
            input_data=train_data,
            targets=None,
            optimizer=optimizer
        )
        
        print(f"Epoch {epoch+1}:")
        for loss_name, loss_value in loss_info.items():
            print(f"  {loss_name}: {loss_value:.4f}")
    
    # Demonstrate enabling/disabling distillation
    print("\nDemonstrating distillation control:")
    print("Disabling distillation for inference...")
    student_model.disable_distillation()
    
    # Test inference without distillation
    with torch.no_grad():
        output = student_model(train_data)
        print(f"Inference output shape: {output.shape}")
    
    print("Re-enabling distillation...")
    student_model.enable_distillation()
    
    # Save model with distillation components
    checkpoint_path = 'output/distillation_checkpoint.pth'
    student_model.save_checkpoint(checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")

if __name__ == "__main__":
    print("PiKV MoE Comprehensive Comparison")
    print("="*60)
    
    # Run standard model comparison
    compare_models()
    
    # Demonstrate distillation functionality
    demonstrate_distillation()
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("Check the 'output/' directory for visualizations and saved models.")
    print("="*60)
