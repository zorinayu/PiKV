#!/usr/bin/env python3
"""
测试MoE策略集成
验证各种MoE路由器是否正常工作
"""

import torch
import sys
import os

# 添加core/single到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'core', 'single'))

from moe_strategies_fixed import create_moe_router

def test_moe_strategies():
    """测试各种MoE策略"""
    print("测试MoE策略集成")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试参数
    hidden_size = 512
    num_experts = 8
    batch_size = 4
    seq_len = 64
    
    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
    print(f"输入数据形状: {hidden_states.shape}")
    
    # 测试策略列表
    strategies = [
        ('base', '基础路由器', {}),
        ('flex', 'Flex-MoE路由器', {
            'modality_info': {
                'image': torch.randn(batch_size, seq_len, hidden_size, device=device),
                'genomic': torch.randn(batch_size, seq_len, hidden_size, device=device)
            }
        }),
        ('time', 'Time-MoE路由器', {
            'time_info': {
                'timestamps': torch.arange(seq_len, device=device).float(),
                'seasonality': torch.sin(torch.arange(seq_len, device=device) * 2 * torch.pi / 24)
            }
        }),
        ('fast', 'FastMoE路由器', {}),
        ('mixture', 'Mixture of Experts路由器', {'training': True})
    ]
    
    # 测试每种策略
    for strategy_name, strategy_desc, extra_args in strategies:
        print(f"\n测试 {strategy_desc}...")
        
        try:
            # 创建路由器
            router = create_moe_router(
                strategy_name, 
                hidden_size=hidden_size, 
                num_experts=num_experts, 
                top_k=2
            ).to(device)
            
            # 前向传播
            if strategy_name == 'flex':
                dispatch, combine, probs, loss = router(hidden_states, **extra_args)
            elif strategy_name == 'time':
                dispatch, combine, probs, loss = router(hidden_states, **extra_args)
            elif strategy_name == 'mixture':
                dispatch, combine, probs, loss = router(hidden_states, training=extra_args['training'])
            else:
                dispatch, combine, probs, loss = router(hidden_states)
            
            # 检查输出形状
            expected_shape = (batch_size, seq_len, num_experts, 1)  # 简化的容量计算
            assert dispatch.shape[:3] == expected_shape[:3], f"Dispatch shape mismatch: {dispatch.shape}"
            assert combine.shape[:3] == expected_shape[:3], f"Combine shape mismatch: {combine.shape}"
            assert probs.shape == (batch_size, seq_len, num_experts), f"Probs shape mismatch: {probs.shape}"
            assert isinstance(loss, float), f"Loss should be float, got {type(loss)}"
            
            print(f"  ✓ 成功 - dispatch: {dispatch.shape}, combine: {combine.shape}, probs: {probs.shape}, loss: {loss:.4f}")
            
        except Exception as e:
            print(f"  ✗ 失败 - {e}")
    
    print("\n" + "=" * 50)
    print("测试完成！")

def test_integration_with_pikv():
    """测试与PiKV的集成"""
    print("\n测试与PiKV的集成")
    print("=" * 50)
    
    try:
        from pikv_moe import PiKVMoE
        from config import config
        
        # 更新配置
        config['hidden_size'] = 512
        config['num_experts'] = 8
        
        # 创建PiKV MoE
        pikv_moe = PiKVMoE(rank=4, alpha=1.0)
        
        # 创建MoE路由器
        router = create_moe_router('flex', hidden_size=512, num_experts=8, top_k=2)
        
        # 测试数据
        input_data = torch.randn(4, 64, 512)
        
        # 测试PiKV MoE
        pikv_output = pikv_moe(input_data)
        print(f"PiKV MoE输出形状: {pikv_output.shape}")
        
        # 测试MoE路由器
        modality_info = {
            'image': torch.randn(4, 64, 512),
            'genomic': torch.randn(4, 64, 512)
        }
        dispatch, combine, probs, loss = router(input_data, modality_info)
        print(f"MoE路由器输出 - dispatch: {dispatch.shape}, loss: {loss:.4f}")
        
        print("  ✓ PiKV与MoE策略集成成功！")
        
    except Exception as e:
        print(f"  ✗ 集成测试失败 - {e}")

if __name__ == "__main__":
    test_moe_strategies()
    test_integration_with_pikv()
