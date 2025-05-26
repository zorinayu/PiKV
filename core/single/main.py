import torch
import os
from config import config
from pikv_moe import PiKVMoE
from normal_moe import StandardMoE
from lora import LoRAPiKVMoE
from distillation import create_teacher_model, distillation_training_step
from cache_scheduling import SchedulingPolicy
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

def demonstrate_cache_scheduling():
    """
    演示缓存调度策略功能
    """
    print("\n" + "="*60)
    print("Cache Scheduling Demonstration")
    print("="*60)
    
    # 创建测试数据
    batch_size = 4
    seq_len = 32
    input_data = generate_data(batch_size, config['hidden_size']).to(config['device'])
    
    # 测试不同的调度策略
    scheduling_policies = [
        SchedulingPolicy.NONE,
        SchedulingPolicy.LRU,
        SchedulingPolicy.H2O,
        SchedulingPolicy.STREAMING_LLM,
        SchedulingPolicy.QUEST,
        SchedulingPolicy.FLEXGEN,
        SchedulingPolicy.LRU_PLUS
    ]
    
    results = {}
    
    for policy in scheduling_policies:
        print(f"\n测试调度策略: {policy.value}")
        print("-" * 40)
        
        # 创建模型
        use_scheduling = policy != SchedulingPolicy.NONE
        model = PiKVMoE(
            rank=4,
            alpha=1.0,
            use_cache_scheduling=use_scheduling,
            cache_scheduling_policy=policy
        ).to(config['device'])
        
        # 进行多次前向传播以测试缓存行为
        total_time = 0
        for i in range(5):
            if torch.cuda.is_available():
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record(torch.cuda.current_stream())
                
                with torch.no_grad():
                    output = model(input_data)
                
                end_time.record(torch.cuda.current_stream())
                torch.cuda.synchronize()
                iteration_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                total_time += iteration_time
            else:
                import time
                start_time_cpu = time.time()
                
                with torch.no_grad():
                    output = model(input_data)
                
                end_time_cpu = time.time()
                iteration_time = end_time_cpu - start_time_cpu
                total_time += iteration_time
        
        avg_time = total_time / 5
        
        # 获取缓存统计信息
        if use_scheduling:
            cache_stats = model.get_cache_stats()
            print(f"平均前向传播时间: {avg_time:.4f}s")
            print(f"输出形状: {output.shape}")
            print("缓存统计信息:")
            for expert_name, stats in cache_stats.items():
                print(f"  {expert_name}: 利用率 {stats['cache_utilization']:.2%}, 策略 {stats['policy']}")
        else:
            print(f"平均前向传播时间: {avg_time:.4f}s")
            print(f"输出形状: {output.shape}")
            print("无缓存调度")
        
        results[policy.value] = {
            'avg_time': avg_time,
            'output_shape': output.shape,
            'cache_stats': cache_stats if use_scheduling else None
        }
    
    return results

def demonstrate_dynamic_scheduling():
    """
    演示动态调度策略切换
    """
    print("\n" + "="*60)
    print("Dynamic Cache Scheduling Demonstration")
    print("="*60)
    
    # 创建模型（初始不启用调度）
    model = PiKVMoE(
        rank=4,
        alpha=1.0,
        use_cache_scheduling=False
    ).to(config['device'])
    
    # 生成测试数据
    input_data = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    
    print("1. 初始状态（无调度）:")
    with torch.no_grad():
        output1 = model(input_data)
    print(f"   输出形状: {output1.shape}")
    
    print("\n2. 启用LRU调度:")
    model.enable_cache_scheduling(SchedulingPolicy.LRU)
    with torch.no_grad():
        output2 = model(input_data)
    print(f"   输出形状: {output2.shape}")
    model.print_cache_stats()
    
    print("\n3. 切换到H2O调度:")
    model.change_cache_scheduling_policy(SchedulingPolicy.H2O)
    with torch.no_grad():
        output3 = model(input_data)
    print(f"   输出形状: {output3.shape}")
    model.print_cache_stats()
    
    print("\n4. 切换到StreamingLLM调度:")
    model.change_cache_scheduling_policy(SchedulingPolicy.STREAMING_LLM)
    with torch.no_grad():
        output4 = model(input_data)
    print(f"   输出形状: {output4.shape}")
    model.print_cache_stats()
    
    print("\n5. 禁用调度:")
    model.disable_cache_scheduling()
    with torch.no_grad():
        output5 = model(input_data)
    print(f"   输出形状: {output5.shape}")
    
    return {
        'no_scheduling': output1.shape,
        'lru_scheduling': output2.shape,
        'h2o_scheduling': output3.shape,
        'streaming_scheduling': output4.shape,
        'disabled_scheduling': output5.shape
    }

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
    print("PiKV MoE Comprehensive Comparison with Cache Scheduling")
    print("="*60)
    
    # Run standard model comparison
    compare_models()
    
    # Demonstrate cache scheduling functionality
    cache_scheduling_results = demonstrate_cache_scheduling()
    
    # Demonstrate dynamic scheduling
    dynamic_results = demonstrate_dynamic_scheduling()
    
    # Demonstrate distillation functionality
    demonstrate_distillation()
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("Check the 'output/' directory for visualizations and saved models.")
    
    # Print cache scheduling summary
    print("\nCache Scheduling Performance Summary:")
    print("-" * 40)
    for policy, results in cache_scheduling_results.items():
        print(f"{policy}: {results['avg_time']:.4f}s average")
    
    print("="*60)
