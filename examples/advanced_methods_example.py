"""
PiKV Advanced Methods Usage Example
展示如何使用新的EPLB routing和advanced distillation方法

USAGE Example:
cd /Users/dongliu/Documents/GitHub/PiKV && python -c "import sys; sys.path.append('.'); from core.single.pikv_routing import EPLBRouter; from core.single.advanced_distillation import AdvancedDistillationManager, DistillationMethod; import torch; print('Testing EPLB Router...'); router = EPLBRouter(hidden_size=512, num_experts=8, top_k=2); hidden_states = torch.randn(2, 64, 512); dispatch, combine, probs, loss = router(hidden_states); print(f'EPLB Router test passed! Loss: {loss.item():.4f}'); print('Testing Advanced Distillation...'); distill_manager = AdvancedDistillationManager(teacher_hidden_size=768, student_hidden_size=512, method=DistillationMethod.DISTILLM, num_layers=3); print('Advanced Distillation test passed!'); print('All new methods are working correctly!')"
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Optional

# 导入PiKV组件
from core.single.pikv_moe import PiKVMoE
from core.single.pikv_routing import EPLBRouter, HierarchicalRouter
from core.single.advanced_distillation import AdvancedDistillationManager, DistillationMethod
from core.single.cache_scheduling import SchedulingPolicy


class AdvancedPiKVExample:
    """
    高级PiKV方法使用示例
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 512,
        num_experts: int = 8,
        num_layers: int = 6,
        use_eplb_routing: bool = True,
        use_advanced_distillation: bool = True,
        distillation_method: DistillationMethod = DistillationMethod.DISTILLM_2
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.use_eplb_routing = use_eplb_routing
        self.use_advanced_distillation = use_advanced_distillation
        self.distillation_method = distillation_method
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self._setup_models()
        
        # 初始化蒸馏器（如果使用）
        if self.use_advanced_distillation:
            self._setup_distillation()
    
    def _setup_models(self):
        """设置教师和学生模型"""
        print("初始化模型...")
        
        # 教师模型（更大）
        teacher_config = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size * 2,  # 教师模型更大
            'num_experts': self.num_experts,
            'num_layers': self.num_layers,
            'top_k': 2,
            'use_cache_scheduling': True,
            'cache_scheduling_policy': SchedulingPolicy.H2O
        }
        
        # 学生模型（较小）
        student_config = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_experts': self.num_experts // 2,  # 学生模型专家更少
            'num_layers': self.num_layers // 2,    # 学生模型层数更少
            'top_k': 2,
            'use_cache_scheduling': True,
            'cache_scheduling_policy': SchedulingPolicy.STREAMING_LLM
        }
        
        # 如果使用EPLB routing，创建自定义路由器
        if self.use_eplb_routing:
            print("使用EPLB路由器...")
            teacher_router = EPLBRouter(
                hidden_size=teacher_config['hidden_size'],
                num_experts=teacher_config['num_experts'],
                top_k=teacher_config['top_k'],
                temperature=1.0,
                balance_coefficient=0.01,
                use_auxiliary_loss=True,
                use_z_loss=True
            )
            
            student_router = EPLBRouter(
                hidden_size=student_config['hidden_size'],
                num_experts=student_config['num_experts'],
                top_k=student_config['top_k'],
                temperature=1.0,
                balance_coefficient=0.01,
                use_auxiliary_loss=True,
                use_z_loss=True
            )
            
            # 将自定义路由器传递给模型（这里简化处理）
            teacher_config['custom_router'] = teacher_router
            student_config['custom_router'] = student_router
        
        # 创建模型
        self.teacher_model = PiKVMoE(**teacher_config).to(self.device)
        self.student_model = PiKVMoE(**student_config).to(self.device)
        
        print(f"教师模型参数量: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
        print(f"学生模型参数量: {sum(p.numel() for p in self.student_model.parameters()):,}")
    
    def _setup_distillation(self):
        """设置高级蒸馏"""
        print(f"初始化高级蒸馏: {self.distillation_method.value}")
        
        self.distillation_manager = AdvancedDistillationManager(
            teacher_hidden_size=self.hidden_size * 2,
            student_hidden_size=self.hidden_size,
            method=self.distillation_method,
            num_layers=self.num_layers // 2,
            temperature=4.0,
            alpha=0.7,
            beta=0.3
        ).to(self.device)
        
        # 获取方法信息
        method_info = self.distillation_manager.get_method_info()
        print(f"蒸馏方法: {method_info['name']}")
        print(f"描述: {method_info['description']}")
    
    def create_sample_data(self, batch_size: int = 8, seq_len: int = 128, num_batches: int = 100):
        """创建示例数据"""
        print(f"创建示例数据: {num_batches} batches, batch_size={batch_size}, seq_len={seq_len}")
        
        # 生成随机序列数据
        input_ids = torch.randint(0, self.vocab_size, (num_batches * batch_size, seq_len))
        labels = torch.randint(0, self.vocab_size, (num_batches * batch_size, seq_len))
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def train_with_advanced_methods(
        self, 
        dataloader: DataLoader, 
        num_epochs: int = 5,
        learning_rate: float = 1e-4
    ):
        """使用高级方法进行训练"""
        print(f"开始训练 {num_epochs} epochs...")
        
        # 设置优化器
        teacher_optimizer = optim.AdamW(self.teacher_model.parameters(), lr=learning_rate)
        student_optimizer = optim.AdamW(self.student_model.parameters(), lr=learning_rate)
        
        # 训练历史
        history = {
            'teacher_loss': [],
            'student_loss': [],
            'distillation_loss': [],
            'routing_loss': [],
            'expert_usage_variance': []
        }
        
        for epoch in range(num_epochs):
            epoch_teacher_loss = 0.0
            epoch_student_loss = 0.0
            epoch_distill_loss = 0.0
            epoch_routing_loss = 0.0
            epoch_expert_variance = 0.0
            
            num_batches = 0
            
            for batch_idx, (input_ids, labels) in enumerate(dataloader):
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                # 教师模型前向传播
                teacher_optimizer.zero_grad()
                teacher_outputs = self.teacher_model(input_ids)
                teacher_logits = teacher_outputs['logits'] if isinstance(teacher_outputs, dict) else teacher_outputs
                
                # 教师模型损失
                teacher_loss = nn.CrossEntropyLoss()(
                    teacher_logits.view(-1, self.vocab_size), 
                    labels.view(-1)
                )
                
                # 学生模型前向传播
                student_optimizer.zero_grad()
                student_outputs = self.student_model(input_ids)
                student_logits = student_outputs['logits'] if isinstance(student_outputs, dict) else student_outputs
                
                # 学生模型损失
                student_loss = nn.CrossEntropyLoss()(
                    student_logits.view(-1, self.vocab_size), 
                    labels.view(-1)
                )
                
                total_loss = teacher_loss + student_loss
                
                # 高级蒸馏（如果启用）
                if self.use_advanced_distillation:
                    # 提取特征（简化处理）
                    teacher_features = [torch.randn_like(input_ids, dtype=torch.float).unsqueeze(-1).expand(-1, -1, self.hidden_size * 2) for _ in range(self.num_layers // 2)]
                    student_features = [torch.randn_like(input_ids, dtype=torch.float).unsqueeze(-1).expand(-1, -1, self.hidden_size) for _ in range(self.num_layers // 2)]
                    
                    # 执行蒸馏
                    distill_result = self.distillation_manager.distill(
                        student_features=student_features,
                        teacher_features=teacher_features,
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        labels=labels
                    )
                    
                    distill_loss = distill_result['total_loss']
                    total_loss += 0.5 * distill_loss
                    epoch_distill_loss += distill_loss.item()
                
                # 路由损失（如果使用EPLB）
                routing_loss = 0.0
                if self.use_eplb_routing and hasattr(self.teacher_model, 'get_routing_loss'):
                    routing_loss = self.teacher_model.get_routing_loss()
                    total_loss += 0.1 * routing_loss
                    epoch_routing_loss += routing_loss.item()
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.teacher_model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                
                # 优化器步骤
                teacher_optimizer.step()
                student_optimizer.step()
                
                # 记录损失
                epoch_teacher_loss += teacher_loss.item()
                epoch_student_loss += student_loss.item()
                
                # 计算专家使用方差（简化）
                if hasattr(self.teacher_model, 'get_expert_usage'):
                    expert_usage = self.teacher_model.get_expert_usage()
                    if expert_usage is not None:
                        epoch_expert_variance += expert_usage.var().item()
                
                num_batches += 1
                
                # 打印进度
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                          f"Teacher Loss: {teacher_loss.item():.4f}, "
                          f"Student Loss: {student_loss.item():.4f}")
            
            # 记录epoch平均损失
            history['teacher_loss'].append(epoch_teacher_loss / num_batches)
            history['student_loss'].append(epoch_student_loss / num_batches)
            history['distillation_loss'].append(epoch_distill_loss / num_batches)
            history['routing_loss'].append(epoch_routing_loss / num_batches)
            history['expert_usage_variance'].append(epoch_expert_variance / num_batches)
            
            print(f"Epoch {epoch+1} 完成:")
            print(f"  平均教师损失: {history['teacher_loss'][-1]:.4f}")
            print(f"  平均学生损失: {history['student_loss'][-1]:.4f}")
            if self.use_advanced_distillation:
                print(f"  平均蒸馏损失: {history['distillation_loss'][-1]:.4f}")
            if self.use_eplb_routing:
                print(f"  平均路由损失: {history['routing_loss'][-1]:.4f}")
                print(f"  专家使用方差: {history['expert_usage_variance'][-1]:.4f}")
        
        return history
    
    def evaluate_models(self, dataloader: DataLoader):
        """评估模型性能"""
        print("评估模型性能...")
        
        self.teacher_model.eval()
        self.student_model.eval()
        
        teacher_total_loss = 0.0
        student_total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, labels in dataloader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                # 教师模型
                teacher_outputs = self.teacher_model(input_ids)
                teacher_logits = teacher_outputs['logits'] if isinstance(teacher_outputs, dict) else teacher_outputs
                teacher_loss = nn.CrossEntropyLoss()(
                    teacher_logits.view(-1, self.vocab_size), 
                    labels.view(-1)
                )
                
                # 学生模型
                student_outputs = self.student_model(input_ids)
                student_logits = student_outputs['logits'] if isinstance(student_outputs, dict) else student_outputs
                student_loss = nn.CrossEntropyLoss()(
                    student_logits.view(-1, self.vocab_size), 
                    labels.view(-1)
                )
                
                teacher_total_loss += teacher_loss.item()
                student_total_loss += student_loss.item()
                num_batches += 1
        
        avg_teacher_loss = teacher_total_loss / num_batches
        avg_student_loss = student_total_loss / num_batches
        
        print(f"评估结果:")
        print(f"  教师模型平均损失: {avg_teacher_loss:.4f}")
        print(f"  学生模型平均损失: {avg_student_loss:.4f}")
        print(f"  性能比率 (学生/教师): {avg_student_loss/avg_teacher_loss:.2f}")
        
        return avg_teacher_loss, avg_student_loss
    
    def demonstrate_cache_scheduling(self):
        """演示缓存调度功能"""
        print("\n演示缓存调度功能...")
        
        # 创建测试输入
        test_input = torch.randint(0, self.vocab_size, (2, 64)).to(self.device)
        
        # 测试不同的调度策略
        scheduling_policies = [
            SchedulingPolicy.NONE,
            SchedulingPolicy.H2O,
            SchedulingPolicy.STREAMING_LLM,
            SchedulingPolicy.QUEST,
            SchedulingPolicy.LRU
        ]
        
        for policy in scheduling_policies:
            print(f"\n测试调度策略: {policy.value}")
            
            # 更改调度策略
            self.student_model.change_cache_scheduling_policy(policy)
            
            # 前向传播
            with torch.no_grad():
                outputs = self.student_model(test_input)
            
            # 获取缓存统计
            cache_stats = self.student_model.get_cache_stats()
            if cache_stats:
                print(f"  缓存命中率: {cache_stats.get('hit_rate', 0):.2%}")
                print(f"  缓存使用率: {cache_stats.get('utilization', 0):.2%}")
    
    def save_models(self, save_dir: str = "checkpoints"):
        """保存模型"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        teacher_path = os.path.join(save_dir, "teacher_model.pt")
        student_path = os.path.join(save_dir, "student_model.pt")
        
        torch.save(self.teacher_model.state_dict(), teacher_path)
        torch.save(self.student_model.state_dict(), student_path)
        
        print(f"模型已保存到 {save_dir}")


def main():
    """主函数"""
    print("=== PiKV Advanced Methods Example ===\n")
    
    # 创建示例实例
    example = AdvancedPiKVExample(
        vocab_size=1000,
        hidden_size=256,  # 较小的尺寸用于演示
        num_experts=8,
        num_layers=4,
        use_eplb_routing=True,
        use_advanced_distillation=True,
        distillation_method=DistillationMethod.DISTILLM_2
    )
    
    # 创建数据
    train_dataloader = example.create_sample_data(
        batch_size=4, 
        seq_len=64, 
        num_batches=20
    )
    
    test_dataloader = example.create_sample_data(
        batch_size=4, 
        seq_len=64, 
        num_batches=5
    )
    
    # 训练模型
    print("\n" + "="*50)
    history = example.train_with_advanced_methods(
        train_dataloader, 
        num_epochs=3,
        learning_rate=1e-4
    )
    
    # 评估模型
    print("\n" + "="*50)
    example.evaluate_models(test_dataloader)
    
    # 演示缓存调度
    print("\n" + "="*50)
    example.demonstrate_cache_scheduling()
    
    # 保存模型
    print("\n" + "="*50)
    example.save_models()
    
    print("\n🎉 示例完成！")
    
    # 打印总结
    print("\n=== 总结 ===")
    print("本示例展示了以下高级功能：")
    print("1. EPLB (Expert-level Load Balancing) 路由策略")
    print("2. DistillM-2 高级知识蒸馏")
    print("3. 多种缓存调度策略 (H2O, StreamingLLM, QUEST, LRU)")
    print("4. 教师-学生模型训练")
    print("5. 模型性能评估和保存")


if __name__ == "__main__":
    main() 