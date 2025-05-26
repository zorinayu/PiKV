#!/usr/bin/env python3
"""
PiKV Cache Scheduling Test Suite
测试各种缓存调度策略的性能和正确性
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple
from .cache_scheduling import (
    SchedulingPolicy, CacheSchedulingManager,
    H2OScheduler, StreamingLLMScheduler, QUESTScheduler,
    FlexGenScheduler, LRUScheduler, LRUPlusScheduler
)
from .pikv_moe import PiKVMoE
from .config import config

class CacheSchedulingTester:
    """缓存调度策略测试器"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.test_results = {}
    
    def generate_test_data(self, batch_size=8, seq_len=128, hidden_size=256):
        """生成测试数据"""
        keys = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        values = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
        importance = torch.rand(batch_size, seq_len, device=self.device)
        attention_weights = torch.rand(batch_size, seq_len, device=self.device)
        
        return keys, values, importance, attention_weights
    
    def test_individual_scheduler(self, scheduler_class, cache_size=512, hidden_size=256, 
                                num_updates=1000, **scheduler_kwargs):
        """测试单个调度器"""
        print(f"\n测试 {scheduler_class.__name__}...")
        
        # 创建调度器
        scheduler = scheduler_class(cache_size, hidden_size, **scheduler_kwargs)
        scheduler = scheduler.to(self.device)
        
        # 生成测试数据
        keys, values, importance, attention_weights = self.generate_test_data(
            batch_size=1, seq_len=num_updates, hidden_size=hidden_size
        )
        
        # 测试性能
        start_time = time.time()
        eviction_counts = []
        
        for i in range(0, num_updates, 32):  # 批量处理
            end_idx = min(i + 32, num_updates)
            batch_keys = keys[:, i:end_idx, :].reshape(-1, hidden_size)
            batch_values = values[:, i:end_idx, :].reshape(-1, hidden_size)
            
            if len(batch_keys) == 0:
                continue
            
            # 模拟缓存更新
            if hasattr(scheduler, 'update_attention_scores'):
                indices = torch.arange(min(len(batch_keys), cache_size), device=self.device)
                scheduler.update_attention_scores(indices, attention_weights[0, i:end_idx][:len(indices)])
            
            # 选择淘汰候选项
            if len(batch_keys) > cache_size:
                evict_mask = scheduler.select_eviction_candidates(
                    batch_keys[:cache_size], batch_values[:cache_size], {}
                )
                eviction_counts.append(evict_mask.sum().item())
        
        end_time = time.time()
        
        # 计算统计信息
        avg_evictions = np.mean(eviction_counts) if eviction_counts else 0
        processing_time = end_time - start_time
        
        results = {
            'scheduler': scheduler_class.__name__,
            'avg_evictions': avg_evictions,
            'processing_time': processing_time,
            'throughput': num_updates / processing_time,
            'eviction_counts': eviction_counts
        }
        
        print(f"  平均淘汰数: {avg_evictions:.2f}")
        print(f"  处理时间: {processing_time:.4f}s")
        print(f"  吞吐量: {results['throughput']:.2f} updates/s")
        
        return results
    
    def test_all_schedulers(self):
        """测试所有调度器"""
        print("="*60)
        print("缓存调度策略性能测试")
        print("="*60)
        
        schedulers_to_test = [
            (H2OScheduler, {}),
            (StreamingLLMScheduler, {}),
            (QUESTScheduler, {}),
            (FlexGenScheduler, {}),
            (LRUScheduler, {}),
            (LRUPlusScheduler, {})
        ]
        
        results = {}
        for scheduler_class, kwargs in schedulers_to_test:
            try:
                result = self.test_individual_scheduler(scheduler_class, **kwargs)
                results[scheduler_class.__name__] = result
            except Exception as e:
                print(f"测试 {scheduler_class.__name__} 时出错: {e}")
                results[scheduler_class.__name__] = None
        
        self.test_results['individual_schedulers'] = results
        return results
    
    def test_cache_scheduling_manager(self):
        """测试缓存调度管理器"""
        print("\n" + "="*60)
        print("缓存调度管理器测试")
        print("="*60)
        
        cache_size = 256
        hidden_size = 128
        
        policies_to_test = [
            SchedulingPolicy.H2O,
            SchedulingPolicy.STREAMING_LLM,
            SchedulingPolicy.QUEST,
            SchedulingPolicy.FLEXGEN,
            SchedulingPolicy.LRU,
            SchedulingPolicy.LRU_PLUS
        ]
        
        results = {}
        
        for policy in policies_to_test:
            print(f"\n测试策略: {policy.value}")
            
            try:
                # 创建管理器
                manager = CacheSchedulingManager(cache_size, hidden_size, policy)
                manager = manager.to(self.device)
                
                # 生成测试数据
                keys, values, importance, _ = self.generate_test_data(
                    batch_size=4, seq_len=64, hidden_size=hidden_size
                )
                
                # 测试缓存更新
                start_time = time.time()
                
                for i in range(10):  # 多次更新测试
                    metadata = {
                        'importance': importance,
                        'timestamp': torch.full((keys.size(0), keys.size(1)), i, device=self.device)
                    }
                    manager.update_cache(keys, values, metadata)
                
                end_time = time.time()
                
                # 获取统计信息
                stats = manager.get_cache_stats()
                
                results[policy.value] = {
                    'stats': stats,
                    'update_time': end_time - start_time,
                    'final_cache_size': stats['cache_size']
                }
                
                print(f"  最终缓存大小: {stats['cache_size']}")
                print(f"  缓存利用率: {stats['cache_utilization']:.2%}")
                print(f"  更新时间: {end_time - start_time:.4f}s")
                
                if 'eviction_count' in stats:
                    print(f"  淘汰次数: {stats['eviction_count']}")
                
            except Exception as e:
                print(f"测试策略 {policy.value} 时出错: {e}")
                results[policy.value] = None
        
        self.test_results['manager_test'] = results
        return results
    
    def test_pikv_moe_integration(self):
        """测试PiKV MoE集成"""
        print("\n" + "="*60)
        print("PiKV MoE 缓存调度集成测试")
        print("="*60)
        
        # 测试不同的调度策略
        policies_to_test = [
            SchedulingPolicy.NONE,
            SchedulingPolicy.LRU,
            SchedulingPolicy.H2O,
            SchedulingPolicy.STREAMING_LLM
        ]
        
        results = {}
        
        for policy in policies_to_test:
            print(f"\n测试策略: {policy.value}")
            
            try:
                # 创建模型
                use_scheduling = policy != SchedulingPolicy.NONE
                model = PiKVMoE(
                    rank=4,
                    alpha=1.0,
                    use_cache_scheduling=use_scheduling,
                    cache_scheduling_policy=policy
                ).to(self.device)
                
                # 生成测试输入
                batch_size = 4
                seq_len = 32
                input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=self.device)
                
                # 前向传播测试
                start_time = time.time()
                
                with torch.no_grad():
                    for i in range(5):  # 多次前向传播
                        output = model(input_ids)
                
                end_time = time.time()
                
                # 获取缓存统计
                if use_scheduling:
                    cache_stats = model.get_cache_stats()
                else:
                    cache_stats = {'policy': 'none'}
                
                results[policy.value] = {
                    'forward_time': end_time - start_time,
                    'output_shape': output.shape,
                    'cache_stats': cache_stats
                }
                
                print(f"  前向传播时间: {end_time - start_time:.4f}s")
                print(f"  输出形状: {output.shape}")
                
                if use_scheduling:
                    print(f"  缓存统计: {len(cache_stats)} 个专家")
                    # 打印缓存统计
                    model.print_cache_stats()
                
            except Exception as e:
                print(f"测试策略 {policy.value} 时出错: {e}")
                results[policy.value] = None
        
        self.test_results['moe_integration'] = results
        return results
    
    def test_dynamic_policy_switching(self):
        """测试动态策略切换"""
        print("\n" + "="*60)
        print("动态策略切换测试")
        print("="*60)
        
        # 创建模型
        model = PiKVMoE(
            rank=4,
            alpha=1.0,
            use_cache_scheduling=False  # 初始禁用
        ).to(self.device)
        
        # 生成测试数据
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=self.device)
        
        print("初始状态 (无调度):")
        with torch.no_grad():
            output1 = model(input_ids)
        print(f"  输出形状: {output1.shape}")
        
        # 启用LRU调度
        print("\n启用LRU调度:")
        model.enable_cache_scheduling(SchedulingPolicy.LRU)
        with torch.no_grad():
            output2 = model(input_ids)
        print(f"  输出形状: {output2.shape}")
        model.print_cache_stats()
        
        # 切换到H2O调度
        print("\n切换到H2O调度:")
        model.change_cache_scheduling_policy(SchedulingPolicy.H2O)
        with torch.no_grad():
            output3 = model(input_ids)
        print(f"  输出形状: {output3.shape}")
        model.print_cache_stats()
        
        # 禁用调度
        print("\n禁用调度:")
        model.disable_cache_scheduling()
        with torch.no_grad():
            output4 = model(input_ids)
        print(f"  输出形状: {output4.shape}")
        
        return {
            'no_scheduling': output1.shape,
            'lru_scheduling': output2.shape,
            'h2o_scheduling': output3.shape,
            'disabled_scheduling': output4.shape
        }
    
    def plot_results(self, save_path='cache_scheduling_results.png'):
        """绘制测试结果"""
        if not self.test_results:
            print("没有测试结果可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PiKV Cache Scheduling Performance Results', fontsize=16)
        
        # 1. 调度器性能比较
        if 'individual_schedulers' in self.test_results:
            scheduler_results = self.test_results['individual_schedulers']
            schedulers = []
            throughputs = []
            
            for name, result in scheduler_results.items():
                if result is not None:
                    schedulers.append(name.replace('Scheduler', ''))
                    throughputs.append(result['throughput'])
            
            axes[0, 0].bar(schedulers, throughputs)
            axes[0, 0].set_title('Scheduler Throughput Comparison')
            axes[0, 0].set_ylabel('Updates/second')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 缓存利用率
        if 'manager_test' in self.test_results:
            manager_results = self.test_results['manager_test']
            policies = []
            utilizations = []
            
            for policy, result in manager_results.items():
                if result is not None and 'stats' in result:
                    policies.append(policy)
                    utilizations.append(result['stats']['cache_utilization'])
            
            axes[0, 1].bar(policies, utilizations)
            axes[0, 1].set_title('Cache Utilization by Policy')
            axes[0, 1].set_ylabel('Utilization Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. MoE集成性能
        if 'moe_integration' in self.test_results:
            moe_results = self.test_results['moe_integration']
            policies = []
            times = []
            
            for policy, result in moe_results.items():
                if result is not None:
                    policies.append(policy)
                    times.append(result['forward_time'])
            
            axes[1, 0].bar(policies, times)
            axes[1, 0].set_title('MoE Forward Pass Time by Policy')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 淘汰次数分布
        if 'individual_schedulers' in self.test_results:
            scheduler_results = self.test_results['individual_schedulers']
            
            for name, result in scheduler_results.items():
                if result is not None and result['eviction_counts']:
                    axes[1, 1].plot(result['eviction_counts'], 
                                   label=name.replace('Scheduler', ''), alpha=0.7)
            
            axes[1, 1].set_title('Eviction Counts Over Time')
            axes[1, 1].set_xlabel('Update Batch')
            axes[1, 1].set_ylabel('Evictions')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n结果图表已保存到: {save_path}")
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("开始PiKV缓存调度策略综合测试...")
        
        # 运行所有测试
        self.test_all_schedulers()
        self.test_cache_scheduling_manager()
        self.test_pikv_moe_integration()
        self.test_dynamic_policy_switching()
        
        # 绘制结果
        self.plot_results()
        
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
        
        return self.test_results

def main():
    """主测试函数"""
    print("PiKV Cache Scheduling Test Suite")
    print("="*60)
    
    # 创建测试器
    tester = CacheSchedulingTester()
    
    # 运行综合测试
    results = tester.run_comprehensive_test()
    
    # 打印总结
    print("\n测试总结:")
    for test_name, test_results in results.items():
        print(f"\n{test_name}:")
        if isinstance(test_results, dict):
            for key, value in test_results.items():
                if isinstance(value, dict) and 'throughput' in value:
                    print(f"  {key}: {value['throughput']:.2f} updates/s")
                elif key and value is not None:
                    print(f"  {key}: 测试完成")

if __name__ == "__main__":
    main() 