#!/usr/bin/env python3
"""
PiKV知识蒸馏简单示例
演示如何在PiKV MoE中使用知识蒸馏功能
"""

import torch
import torch.optim as optim
import sys
import os

# 添加core/single到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core', 'single'))

from config import config
from pikv_moe import PiKVMoE
from distillation import create_teacher_model

def simple_distillation_example():
    """
    简单的知识蒸馏示例
    """
    print("PiKV知识蒸馏简单示例")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 创建带知识蒸馏的学生模型
    print("\n1. 创建学生模型（带知识蒸馏）...")
    student_model = PiKVMoE(
        rank=4,                                    # LoRA rank
        alpha=1.0,                                # LoRA alpha
        use_distillation=True,                    # 启用知识蒸馏
        teacher_hidden_size=config['hidden_size'] * 2  # 教师模型更大
    ).to(device)
    
    print(f"学生模型参数数量: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # 2. 生成示例数据
    print("\n2. 生成训练数据...")
    batch_size = 4
    seq_len = 32
    hidden_size = config['hidden_size']
    
    # 输入数据
    input_data = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # 目标数据（用于语言建模）
    targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
    
    print(f"输入数据形状: {input_data.shape}")
    print(f"目标数据形状: {targets.shape}")
    
    # 3. 设置优化器
    print("\n3. 设置优化器...")
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
    
    # 4. 训练循环
    print("\n4. 开始知识蒸馏训练...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        # 执行一步知识蒸馏训练
        loss_info = student_model.distillation_step(
            input_data=input_data,
            targets=targets,
            optimizer=optimizer
        )
        
        # 打印损失信息
        print(f"Epoch {epoch+1}/{num_epochs}:")
        for loss_name, loss_value in loss_info.items():
            print(f"  {loss_name}: {loss_value:.4f}")
    
    # 5. 推理模式
    print("\n5. 切换到推理模式...")
    student_model.disable_distillation()  # 关闭蒸馏以提高推理速度
    student_model.eval()
    
    with torch.no_grad():
        output = student_model(input_data)
        print(f"推理输出形状: {output.shape}")
    
    # 6. 保存模型
    print("\n6. 保存模型...")
    save_path = 'distillation_model.pth'
    student_model.save_checkpoint(save_path)
    print(f"模型已保存到: {save_path}")
    
    print("\n知识蒸馏示例完成！")

def advanced_distillation_example():
    """
    高级知识蒸馏示例 - 展示更多功能
    """
    print("\n" + "=" * 50)
    print("高级知识蒸馏示例")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 创建学生模型
    student_model = PiKVMoE(
        rank=4,
        alpha=1.0,
        use_distillation=True,
        teacher_hidden_size=config['hidden_size'] * 2
    ).to(device)
    
    # 2. 手动创建教师模型（可选）
    print("\n创建自定义教师模型...")
    teacher_model = create_teacher_model(
        hidden_size=config['hidden_size'] * 2,
        num_experts=config['num_experts']
    ).to(device)
    
    print(f"教师模型参数数量: {sum(p.numel() for p in teacher_model.parameters()):,}")
    
    # 3. 生成更复杂的数据
    batch_size = 8
    seq_len = 64
    input_data = torch.randn(batch_size, seq_len, config['hidden_size'], device=device)
    targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
    
    # 4. 自定义蒸馏参数
    print("\n调整蒸馏参数...")
    # 修改温度参数
    student_model.distillation_module.kd_loss.temperature = 6.0
    # 修改损失权重
    student_model.distillation_module.expert_distill_weight = 0.5
    student_model.distillation_module.cache_distill_weight = 0.4
    
    # 5. 训练循环
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
    
    print("\n开始高级蒸馏训练...")
    for epoch in range(3):
        loss_info = student_model.distillation_step(
            input_data=input_data,
            targets=targets,
            optimizer=optimizer
        )
        
        print(f"Epoch {epoch+1}:")
        for loss_name, loss_value in loss_info.items():
            print(f"  {loss_name}: {loss_value:.4f}")
    
    # 6. 演示模型控制
    print("\n演示模型控制功能...")
    
    # 禁用蒸馏
    student_model.disable_distillation()
    print("蒸馏已禁用")
    
    # 重新启用蒸馏
    student_model.enable_distillation()
    print("蒸馏已重新启用")
    
    # 7. 比较有无蒸馏的性能
    print("\n比较有无蒸馏的性能...")
    
    # 不使用蒸馏的模型
    model_no_distill = PiKVMoE(rank=4, alpha=1.0, use_distillation=False).to(device)
    
    # 简单性能测试
    import time
    
    test_data = torch.randn(4, 32, config['hidden_size'], device=device)
    
    # 测试蒸馏模型
    student_model.disable_distillation()  # 推理时关闭蒸馏
    start_time = time.time()
    with torch.no_grad():
        _ = student_model(test_data)
    distill_time = time.time() - start_time
    
    # 测试普通模型
    start_time = time.time()
    with torch.no_grad():
        _ = model_no_distill(test_data)
    normal_time = time.time() - start_time
    
    print(f"蒸馏模型推理时间: {distill_time:.4f}s")
    print(f"普通模型推理时间: {normal_time:.4f}s")
    
    print("\n高级蒸馏示例完成！")

def load_and_use_distilled_model():
    """
    演示如何加载和使用已训练的蒸馏模型
    """
    print("\n" + "=" * 50)
    print("加载和使用蒸馏模型示例")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 创建模型架构
    model = PiKVMoE(
        rank=4,
        alpha=1.0,
        use_distillation=True,
        teacher_hidden_size=config['hidden_size'] * 2
    ).to(device)
    
    # 2. 尝试加载之前保存的模型
    try:
        model.load_checkpoint('distillation_model.pth')
        print("成功加载蒸馏模型")
        
        # 3. 使用模型进行推理
        model.disable_distillation()  # 推理时关闭蒸馏
        model.eval()
        
        test_input = torch.randn(2, 16, config['hidden_size'], device=device)
        
        with torch.no_grad():
            output = model(test_input)
            print(f"推理输出形状: {output.shape}")
            print("模型推理成功！")
            
    except FileNotFoundError:
        print("未找到保存的模型文件，请先运行简单示例")
    except Exception as e:
        print(f"加载模型时出错: {e}")

if __name__ == "__main__":
    # 运行所有示例
    simple_distillation_example()
    advanced_distillation_example()
    load_and_use_distilled_model()
    
    print("\n" + "=" * 50)
    print("所有知识蒸馏示例完成！")
    print("=" * 50) 