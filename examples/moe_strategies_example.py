#!/usr/bin/env python3
"""
MoE策略使用示例
演示如何在PiKV中使用不同的MoE策略
"""

import torch
import torch.nn as nn
import sys
import os

# 添加core/single到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core', 'single'))

from moe_strategies_fixed import create_moe_router

class PiKVWithMoEStrategies(nn.Module):
    """
    集成多种MoE策略的PiKV模型
    """
    def __init__(self, hidden_size=1024, num_experts=8, strategy='flex'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.strategy = strategy
        
        # 输入投影层
        self.input_projection = nn.Linear(hidden_size, hidden_size)
        
        # 创建MoE路由器
        self.router = create_moe_router(
            strategy, 
            hidden_size=hidden_size, 
            num_experts=num_experts, 
            top_k=2
        )
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size)
            ) for _ in range(num_experts)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, **kwargs):
        # 输入投影
        x = self.input_projection(x)
        
        # 路由
        dispatch, combine, probs, aux_loss = self.router(x, **kwargs)
        
        # 专家处理
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 简化的专家处理逻辑
            expert_input = x  # 在实际实现中，这里应该使用dispatch张量
            expert_output = expert(expert_input)
            expert_outputs.append(expert_output)
        
        # 组合专家输出
        combined_output = torch.stack(expert_outputs, dim=2)  # [batch, seq, num_experts, hidden]
        combined_output = combined_output.mean(dim=2)  # [batch, seq, hidden]
        
        # 输出投影
        output = self.output_projection(combined_output)
        
        return output, aux_loss

def example_flex_moe():
    """Flex-MoE示例：多模态学习"""
    print("Flex-MoE示例：多模态学习")
    print("-" * 40)
    
    # 创建模型
    model = PiKVWithMoEStrategies(strategy='flex')
    
    # 多模态数据
    batch_size, seq_len, hidden_size = 4, 128, 1024
    text_data = torch.randn(batch_size, seq_len, hidden_size)
    image_data = torch.randn(batch_size, seq_len, hidden_size)
    genomic_data = torch.randn(batch_size, seq_len, hidden_size)
    
    # 模态信息
    modality_info = {
        'image': image_data,
        'genomic': genomic_data
    }
    
    # 前向传播
    output, aux_loss = model(text_data, modality_info=modality_info)
    print(f"输入形状: {text_data.shape}")
    print(f"输出形状: {output.shape}")
    print(f"辅助损失: {aux_loss:.4f}")
    print()

def example_time_moe():
    """Time-MoE示例：时间序列预测"""
    print("Time-MoE示例：时间序列预测")
    print("-" * 40)
    
    # 创建模型
    model = PiKVWithMoEStrategies(strategy='time')
    
    # 时间序列数据
    batch_size, seq_len, hidden_size = 4, 512, 1024
    timeseries_data = torch.randn(batch_size, seq_len, hidden_size)
    
    # 时间信息
    time_info = {
        'timestamps': torch.arange(seq_len).float(),
        'seasonality': torch.sin(torch.arange(seq_len) * 2 * torch.pi / 24)  # 24小时周期
    }
    
    # 前向传播
    output, aux_loss = model(timeseries_data, time_info=time_info)
    print(f"输入形状: {timeseries_data.shape}")
    print(f"输出形状: {output.shape}")
    print(f"辅助损失: {aux_loss:.4f}")
    print()

def example_fast_moe():
    """FastMoE示例：高性能分布式训练"""
    print("FastMoE示例：高性能分布式训练")
    print("-" * 40)
    
    # 创建模型
    model = PiKVWithMoEStrategies(strategy='fast')
    
    # 标准数据
    batch_size, seq_len, hidden_size = 8, 256, 1024
    input_data = torch.randn(batch_size, seq_len, hidden_size)
    
    # 前向传播
    output, aux_loss = model(input_data)
    print(f"输入形状: {input_data.shape}")
    print(f"输出形状: {output.shape}")
    print(f"辅助损失: {aux_loss:.4f}")
    print()

def example_mixture_of_experts():
    """Mixture of Experts示例：通用MoE架构"""
    print("Mixture of Experts示例：通用MoE架构")
    print("-" * 40)
    
    # 创建模型
    model = PiKVWithMoEStrategies(strategy='mixture')
    
    # 标准数据
    batch_size, seq_len, hidden_size = 4, 128, 1024
    input_data = torch.randn(batch_size, seq_len, hidden_size)
    
    # 训练模式
    model.train()
    output, aux_loss = model(input_data, training=True)
    print(f"训练模式 - 输入形状: {input_data.shape}")
    print(f"训练模式 - 输出形状: {output.shape}")
    print(f"训练模式 - 辅助损失: {aux_loss:.4f}")
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        output, aux_loss = model(input_data, training=False)
    print(f"评估模式 - 输出形状: {output.shape}")
    print(f"评估模式 - 辅助损失: {aux_loss:.4f}")
    print()

def example_strategy_comparison():
    """策略比较示例"""
    print("策略比较示例")
    print("-" * 40)
    
    # 测试数据
    batch_size, seq_len, hidden_size = 4, 64, 512
    input_data = torch.randn(batch_size, seq_len, hidden_size)
    
    # 测试不同策略
    strategies = ['base', 'flex', 'time', 'fast', 'mixture']
    
    for strategy in strategies:
        print(f"测试 {strategy} 策略...")
        
        try:
            # 创建路由器
            router = create_moe_router(strategy, hidden_size, num_experts=8, top_k=2)
            
            # 准备额外参数
            extra_args = {}
            if strategy == 'flex':
                extra_args['modality_info'] = {
                    'image': torch.randn(batch_size, seq_len, hidden_size),
                    'genomic': torch.randn(batch_size, seq_len, hidden_size)
                }
            elif strategy == 'time':
                extra_args['time_info'] = {
                    'timestamps': torch.arange(seq_len).float(),
                    'seasonality': torch.sin(torch.arange(seq_len) * 2 * torch.pi / 24)
                }
            elif strategy == 'mixture':
                extra_args['training'] = True
            
            # 前向传播
            dispatch, combine, probs, loss = router(input_data, **extra_args)
            
            print(f"  ✓ 成功 - 损失: {loss:.4f}")
            
        except Exception as e:
            print(f"  ✗ 失败 - {e}")
    
    print()

def main():
    """主函数"""
    print("PiKV MoE策略使用示例")
    print("=" * 60)
    
    # 运行各种示例
    example_flex_moe()
    example_time_moe()
    example_fast_moe()
    example_mixture_of_experts()
    example_strategy_comparison()
    
    print("所有示例运行完成！")
    print("\n使用说明：")
    print("1. Flex-MoE适用于多模态学习场景")
    print("2. Time-MoE适用于时间序列预测任务")
    print("3. FastMoE适用于大规模分布式训练")
    print("4. Mixture of Experts适用于通用场景")
    print("5. 可以根据具体需求选择合适的策略")

if __name__ == "__main__":
    main()
