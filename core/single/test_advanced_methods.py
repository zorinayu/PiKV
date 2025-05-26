"""
测试新的高级方法：EPLB Routing 和 Advanced Distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

# 导入新的方法
from pikv_routing import EPLBRouter, HierarchicalRouter
from advanced_distillation import (
    AdvancedDistillationManager, 
    DistillationMethod,
    DistillMDistillation,
    DistillM2Distillation,
    SpeculativeKDDistillation
)


def test_eplb_router():
    """测试EPLB路由器"""
    print("=== 测试 EPLB Router ===")
    
    # 参数设置
    batch_size = 4
    seq_len = 128
    hidden_size = 512
    num_experts = 8
    top_k = 2
    
    # 创建EPLB路由器
    router = EPLBRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        temperature=1.0,
        balance_coefficient=0.01,
        use_auxiliary_loss=True,
        use_z_loss=True
    )
    
    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 前向传播
    dispatch_tensor, combine_tensor, router_probs, aux_loss = router(hidden_states)
    
    print(f"输入形状: {hidden_states.shape}")
    print(f"调度张量形状: {dispatch_tensor.shape}")
    print(f"组合张量形状: {combine_tensor.shape}")
    print(f"路由概率形状: {router_probs.shape}")
    print(f"辅助损失: {aux_loss.item():.4f}")
    
    # 分析专家使用分布
    expert_usage = router_probs.mean(dim=[0, 1])
    print(f"专家使用分布: {expert_usage}")
    print(f"专家使用方差: {expert_usage.var().item():.4f}")
    
    # 测试多次前向传播，观察负载平衡效果
    usage_history = []
    for i in range(10):
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        _, _, router_probs, _ = router(hidden_states)
        expert_usage = router_probs.mean(dim=[0, 1])
        usage_history.append(expert_usage.detach().numpy())
    
    # 可视化专家使用趋势
    usage_history = np.array(usage_history)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    for i in range(num_experts):
        plt.plot(usage_history[:, i], label=f'Expert {i}')
    plt.title('Expert Usage Over Time (EPLB)')
    plt.xlabel('Iteration')
    plt.ylabel('Usage Probability')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    final_usage = usage_history[-1]
    plt.bar(range(num_experts), final_usage)
    plt.title('Final Expert Usage Distribution')
    plt.xlabel('Expert ID')
    plt.ylabel('Usage Probability')
    plt.axhline(y=1/num_experts, color='r', linestyle='--', label='Uniform')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('eplb_router_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("EPLB Router 测试完成！")


def test_hierarchical_router():
    """测试分层路由器"""
    print("\n=== 测试 Hierarchical Router ===")
    
    # 参数设置
    batch_size = 4
    seq_len = 128
    hidden_size = 512
    num_experts = 16  # 更多专家以展示分层效果
    num_groups = 4
    top_k = 2
    
    # 创建分层路由器
    router = HierarchicalRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        num_groups=num_groups,
        group_top_k=1
    )
    
    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 前向传播
    dispatch_tensor, combine_tensor, router_probs, aux_loss = router(hidden_states)
    
    print(f"输入形状: {hidden_states.shape}")
    print(f"专家总数: {num_experts}, 组数: {num_groups}, 每组专家数: {num_experts // num_groups}")
    print(f"调度张量形状: {dispatch_tensor.shape}")
    print(f"组合张量形状: {combine_tensor.shape}")
    print(f"路由概率形状: {router_probs.shape}")
    print(f"辅助损失: {aux_loss.item():.4f}")
    
    # 分析组级和专家级使用分布
    expert_usage = router_probs.mean(dim=[0, 1])
    group_usage = expert_usage.view(num_groups, -1).sum(dim=1)
    
    print(f"组使用分布: {group_usage}")
    print(f"专家使用分布: {expert_usage}")
    
    # 可视化分层使用分布
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(num_groups), group_usage.detach().numpy())
    plt.title('Group Usage Distribution')
    plt.xlabel('Group ID')
    plt.ylabel('Usage Probability')
    plt.axhline(y=1/num_groups, color='r', linestyle='--', label='Uniform')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    expert_usage_np = expert_usage.detach().numpy()
    colors = ['C0', 'C1', 'C2', 'C3'] * (num_experts // num_groups)
    plt.bar(range(num_experts), expert_usage_np, color=colors)
    plt.title('Expert Usage Distribution (Colored by Group)')
    plt.xlabel('Expert ID')
    plt.ylabel('Usage Probability')
    plt.axhline(y=1/num_experts, color='r', linestyle='--', label='Uniform')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hierarchical_router_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Hierarchical Router 测试完成！")


def test_advanced_distillation():
    """测试高级蒸馏方法"""
    print("\n=== 测试 Advanced Distillation Methods ===")
    
    # 参数设置
    batch_size = 4
    seq_len = 128
    teacher_hidden_size = 768
    student_hidden_size = 512
    vocab_size = 1000
    num_layers = 3
    
    # 创建模拟的教师和学生特征
    teacher_features = [
        torch.randn(batch_size, seq_len, teacher_hidden_size) 
        for _ in range(num_layers)
    ]
    student_features = [
        torch.randn(batch_size, seq_len, student_hidden_size) 
        for _ in range(num_layers)
    ]
    
    # 创建模拟的logits
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 测试不同的蒸馏方法
    methods = [
        DistillationMethod.DISTILLM,
        DistillationMethod.DISTILLM_2,
        DistillationMethod.SPECULATIVE_KD
    ]
    
    results = {}
    
    for method in methods:
        print(f"\n--- 测试 {method.value} ---")
        
        # 创建蒸馏管理器
        distill_manager = AdvancedDistillationManager(
            teacher_hidden_size=teacher_hidden_size,
            student_hidden_size=student_hidden_size,
            method=method,
            num_layers=num_layers
        )
        
        # 执行蒸馏
        loss_dict = distill_manager.distill(
            student_features=student_features,
            teacher_features=teacher_features,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels
        )
        
        # 记录结果
        results[method.value] = loss_dict
        
        # 打印损失信息
        print(f"总损失: {loss_dict['total_loss'].item():.4f}")
        print(f"蒸馏损失: {loss_dict['distill_loss'].item():.4f}")
        print(f"KL损失: {loss_dict['kl_loss'].item():.4f}")
        
        # 打印方法特有的损失
        if 'feature_loss' in loss_dict:
            print(f"特征损失: {loss_dict['feature_loss'].item():.4f}")
        if 'discriminative_loss' in loss_dict:
            print(f"判别性损失: {loss_dict['discriminative_loss'].item():.4f}")
        if 'multi_scale_loss' in loss_dict:
            print(f"多尺度损失: {loss_dict['multi_scale_loss'].item():.4f}")
        if 'attention_loss' in loss_dict:
            print(f"注意力损失: {loss_dict['attention_loss'].item():.4f}")
        if 'speculation_loss' in loss_dict:
            print(f"投机损失: {loss_dict['speculation_loss'].item():.4f}")
        if 'verification_loss' in loss_dict:
            print(f"验证损失: {loss_dict['verification_loss'].item():.4f}")
        if 'prediction_accuracy' in loss_dict:
            print(f"预测准确率: {loss_dict['prediction_accuracy'].item():.4f}")
        
        # 获取方法信息
        method_info = distill_manager.get_method_info()
        print(f"方法描述: {method_info['description']}")
    
    # 可视化不同方法的损失比较
    plt.figure(figsize=(15, 10))
    
    # 总损失比较
    plt.subplot(2, 3, 1)
    methods_names = list(results.keys())
    total_losses = [results[method]['total_loss'].item() for method in methods_names]
    plt.bar(methods_names, total_losses)
    plt.title('Total Loss Comparison')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 蒸馏损失比较
    plt.subplot(2, 3, 2)
    distill_losses = [results[method]['distill_loss'].item() for method in methods_names]
    plt.bar(methods_names, distill_losses)
    plt.title('Distillation Loss Comparison')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # KL损失比较
    plt.subplot(2, 3, 3)
    kl_losses = [results[method]['kl_loss'].item() for method in methods_names]
    plt.bar(methods_names, kl_losses)
    plt.title('KL Loss Comparison')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 特征相关损失
    plt.subplot(2, 3, 4)
    feature_losses = []
    feature_methods = []
    for method in methods_names:
        if 'feature_loss' in results[method]:
            feature_losses.append(results[method]['feature_loss'].item())
            feature_methods.append(method)
        elif 'multi_scale_loss' in results[method]:
            feature_losses.append(results[method]['multi_scale_loss'].item())
            feature_methods.append(method)
    
    if feature_losses:
        plt.bar(feature_methods, feature_losses)
        plt.title('Feature-related Loss')
        plt.ylabel('Loss')
        plt.xticks(rotation=45)
        plt.grid(True)
    
    # 方法特有损失
    plt.subplot(2, 3, 5)
    special_losses = []
    special_methods = []
    special_labels = []
    
    for method in methods_names:
        if 'discriminative_loss' in results[method]:
            special_losses.append(results[method]['discriminative_loss'].item())
            special_methods.append(method)
            special_labels.append('Discriminative')
        elif 'speculation_loss' in results[method]:
            special_losses.append(results[method]['speculation_loss'].item())
            special_methods.append(method)
            special_labels.append('Speculation')
        elif 'attention_loss' in results[method]:
            special_losses.append(results[method]['attention_loss'].item())
            special_methods.append(method)
            special_labels.append('Attention')
    
    if special_losses:
        bars = plt.bar(special_methods, special_losses)
        plt.title('Method-specific Losses')
        plt.ylabel('Loss')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # 添加标签
        for bar, label in zip(bars, special_labels):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    label, ha='center', va='bottom', fontsize=8)
    
    # 预测准确率（如果有）
    plt.subplot(2, 3, 6)
    accuracies = []
    acc_methods = []
    for method in methods_names:
        if 'prediction_accuracy' in results[method]:
            accuracies.append(results[method]['prediction_accuracy'].item())
            acc_methods.append(method)
    
    if accuracies:
        plt.bar(acc_methods, accuracies)
        plt.title('Prediction Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('advanced_distillation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nAdvanced Distillation 测试完成！")


def test_integration():
    """测试EPLB路由器与高级蒸馏的集成"""
    print("\n=== 测试 EPLB + Advanced Distillation 集成 ===")
    
    # 参数设置
    batch_size = 4
    seq_len = 64
    hidden_size = 512
    num_experts = 8
    top_k = 2
    vocab_size = 1000
    
    # 创建EPLB路由器
    router = EPLBRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        temperature=1.0,
        balance_coefficient=0.01
    )
    
    # 创建高级蒸馏管理器
    distill_manager = AdvancedDistillationManager(
        teacher_hidden_size=hidden_size,
        student_hidden_size=hidden_size,
        method=DistillationMethod.DISTILLM_2,
        num_layers=2
    )
    
    # 模拟训练循环
    print("模拟训练循环...")
    routing_losses = []
    distill_losses = []
    expert_usage_variance = []
    
    for epoch in range(10):
        # 创建批次数据
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        teacher_features = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(2)]
        student_features = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(2)]
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # 路由计算
        _, _, router_probs, routing_loss = router(hidden_states)
        
        # 蒸馏计算
        distill_result = distill_manager.distill(
            student_features=student_features,
            teacher_features=teacher_features,
            student_logits=student_logits,
            teacher_logits=teacher_logits
        )
        
        # 记录指标
        routing_losses.append(routing_loss.item())
        distill_losses.append(distill_result['total_loss'].item())
        
        expert_usage = router_probs.mean(dim=[0, 1])
        expert_usage_variance.append(expert_usage.var().item())
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Routing Loss = {routing_loss.item():.4f}, "
                  f"Distill Loss = {distill_result['total_loss'].item():.4f}, "
                  f"Expert Variance = {expert_usage.var().item():.4f}")
    
    # 可视化集成训练过程
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(routing_losses, 'b-', label='Routing Loss')
    plt.title('Routing Loss Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(distill_losses, 'r-', label='Distillation Loss')
    plt.title('Distillation Loss Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(expert_usage_variance, 'g-', label='Expert Usage Variance')
    plt.title('Expert Load Balancing Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('integration_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("集成测试完成！")


def main():
    """主测试函数"""
    print("开始测试新的高级方法...")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 测试EPLB路由器
        test_eplb_router()
        
        # 测试分层路由器
        test_hierarchical_router()
        
        # 测试高级蒸馏方法
        test_advanced_distillation()
        
        # 测试集成
        test_integration()
        
        print("\n🎉 所有测试完成！")
        print("\n生成的图片文件：")
        print("- eplb_router_analysis.png")
        print("- hierarchical_router_analysis.png") 
        print("- advanced_distillation_comparison.png")
        print("- integration_training.png")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 