#!/usr/bin/env python3
"""
PiKV Cache Scheduling Usage Example
简单的缓存调度策略使用示例
"""

import torch
from pikv_moe import PiKVMoE
from cache_scheduling import SchedulingPolicy
from config import config

def basic_usage_example():
    """基本使用示例"""
    print("=== PiKV Cache Scheduling Basic Usage ===\n")
    
    # 1. 创建不带缓存调度的模型（默认行为）
    print("1. 创建标准PiKV模型（无缓存调度）:")
    model_standard = PiKVMoE(rank=4, alpha=1.0)
    print("   ✓ 标准模型已创建")
    
    # 2. 创建带LRU缓存调度的模型
    print("\n2. 创建带LRU缓存调度的PiKV模型:")
    model_lru = PiKVMoE(
        rank=4, 
        alpha=1.0,
        use_cache_scheduling=True,
        cache_scheduling_policy=SchedulingPolicy.LRU
    )
    print("   ✓ LRU调度模型已创建")
    
    # 3. 创建带H2O缓存调度的模型
    print("\n3. 创建带H2O缓存调度的PiKV模型:")
    model_h2o = PiKVMoE(
        rank=4,
        alpha=1.0,
        use_cache_scheduling=True,
        cache_scheduling_policy=SchedulingPolicy.H2O
    )
    print("   ✓ H2O调度模型已创建")
    
    # 4. 测试模型推理
    print("\n4. 测试模型推理:")
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    with torch.no_grad():
        output_standard = model_standard(input_ids)
        output_lru = model_lru(input_ids)
        output_h2o = model_h2o(input_ids)
    
    print(f"   标准模型输出形状: {output_standard.shape}")
    print(f"   LRU模型输出形状: {output_lru.shape}")
    print(f"   H2O模型输出形状: {output_h2o.shape}")
    
    # 5. 查看缓存统计
    print("\n5. 缓存统计信息:")
    print("   LRU模型缓存统计:")
    model_lru.print_cache_stats()
    
    print("   H2O模型缓存统计:")
    model_h2o.print_cache_stats()

def dynamic_scheduling_example():
    """动态调度策略切换示例"""
    print("\n=== Dynamic Scheduling Example ===\n")
    
    # 创建模型（初始不启用调度）
    model = PiKVMoE(rank=4, alpha=1.0, use_cache_scheduling=False)
    input_ids = torch.randint(0, config['vocab_size'], (2, 16))
    
    print("1. 初始状态（无调度）:")
    with torch.no_grad():
        output = model(input_ids)
    print(f"   输出形状: {output.shape}")
    
    print("\n2. 动态启用LRU调度:")
    model.enable_cache_scheduling(SchedulingPolicy.LRU)
    with torch.no_grad():
        output = model(input_ids)
    print(f"   输出形状: {output.shape}")
    
    print("\n3. 切换到H2O调度:")
    model.change_cache_scheduling_policy(SchedulingPolicy.H2O)
    with torch.no_grad():
        output = model(input_ids)
    print(f"   输出形状: {output.shape}")
    
    print("\n4. 切换到StreamingLLM调度:")
    model.change_cache_scheduling_policy(SchedulingPolicy.STREAMING_LLM)
    with torch.no_grad():
        output = model(input_ids)
    print(f"   输出形状: {output.shape}")
    
    print("\n5. 禁用调度:")
    model.disable_cache_scheduling()
    with torch.no_grad():
        output = model(input_ids)
    print(f"   输出形状: {output.shape}")

def all_policies_example():
    """所有调度策略示例"""
    print("\n=== All Scheduling Policies Example ===\n")
    
    policies = [
        SchedulingPolicy.NONE,
        SchedulingPolicy.LRU,
        SchedulingPolicy.LRU_PLUS,
        SchedulingPolicy.H2O,
        SchedulingPolicy.STREAMING_LLM,
        SchedulingPolicy.QUEST,
        SchedulingPolicy.FLEXGEN
    ]
    
    input_ids = torch.randint(0, config['vocab_size'], (2, 16))
    
    for policy in policies:
        print(f"测试策略: {policy.value}")
        
        use_scheduling = policy != SchedulingPolicy.NONE
        model = PiKVMoE(
            rank=4,
            alpha=1.0,
            use_cache_scheduling=use_scheduling,
            cache_scheduling_policy=policy
        )
        
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"  输出形状: {output.shape}")
        
        if use_scheduling:
            stats = model.get_cache_stats()
            print(f"  专家数量: {len(stats)}")
            for expert_name, expert_stats in stats.items():
                print(f"    {expert_name}: 利用率 {expert_stats['cache_utilization']:.2%}")
        
        print()

def performance_comparison_example():
    """性能比较示例"""
    print("\n=== Performance Comparison Example ===\n")
    
    policies_to_test = [
        SchedulingPolicy.NONE,
        SchedulingPolicy.LRU,
        SchedulingPolicy.H2O,
        SchedulingPolicy.STREAMING_LLM
    ]
    
    input_ids = torch.randint(0, config['vocab_size'], (4, 32))
    results = {}
    
    for policy in policies_to_test:
        print(f"测试策略: {policy.value}")
        
        use_scheduling = policy != SchedulingPolicy.NONE
        model = PiKVMoE(
            rank=4,
            alpha=1.0,
            use_cache_scheduling=use_scheduling,
            cache_scheduling_policy=policy
        )
        
        # 预热
        with torch.no_grad():
            _ = model(input_ids)
        
        # 性能测试
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                output = model(input_ids)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        results[policy.value] = avg_time
        print(f"  平均推理时间: {avg_time:.4f}s")
        print()
    
    # 打印比较结果
    print("性能比较结果:")
    baseline = results['none']
    for policy, time_taken in results.items():
        speedup = baseline / time_taken
        print(f"  {policy}: {time_taken:.4f}s (相对加速: {speedup:.2f}x)")

def main():
    """主函数"""
    print("PiKV Cache Scheduling Usage Examples")
    print("=" * 50)
    
    # 运行所有示例
    basic_usage_example()
    dynamic_scheduling_example()
    all_policies_example()
    performance_comparison_example()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成!")
    print("=" * 50)

if __name__ == "__main__":
    main() 