#!/usr/bin/env python3
"""
PiKV知识蒸馏测试脚本
演示如何在PiKV MoE中使用知识蒸馏功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from typing import Dict, List, Tuple

# 导入PiKV模块
from config import config
from pikv_moe import PiKVMoE
from normal_moe import StandardMoE
from lora import LoRAPiKVMoE
from distillation import create_teacher_model, distillation_training_step
from utils import generate_data

class DistillationTrainer:
    """
    知识蒸馏训练器
    """
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.results = {}
        
        # 创建输出目录
        os.makedirs('distillation_results', exist_ok=True)
    
    def create_models(self) -> Tuple[PiKVMoE, nn.Module]:
        """
        创建学生模型和教师模型
        """
        # 学生模型 - 使用知识蒸馏的PiKV MoE
        student_model = PiKVMoE(
            rank=4, 
            alpha=1.0, 
            use_distillation=True,
            teacher_hidden_size=config['hidden_size'] * 2  # 教师模型更大
        ).to(self.device)
        
        # 教师模型 - 更大的模型
        teacher_model = create_teacher_model(
            hidden_size=config['hidden_size'] * 2,
            num_experts=config['num_experts']
        ).to(self.device)
        
        return student_model, teacher_model
    
    def generate_training_data(self, num_samples=1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成训练数据
        """
        # 生成输入数据
        batch_size = config['batch_size']
        seq_len = 64  # 序列长度
        hidden_size = config['hidden_size']
        
        # 创建多个批次的数据
        all_inputs = []
        all_targets = []
        
        for _ in range(num_samples // batch_size):
            # 输入数据
            inputs = torch.randn(batch_size, seq_len, hidden_size, device=self.device)
            
            # 目标数据（模拟语言建模任务）
            targets = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=self.device)
            
            all_inputs.append(inputs)
            all_targets.append(targets)
        
        return torch.cat(all_inputs, dim=0), torch.cat(all_targets, dim=0)
    
    def train_with_distillation(
        self, 
        student_model: PiKVMoE, 
        teacher_model: nn.Module,
        train_data: torch.Tensor,
        train_targets: torch.Tensor,
        epochs: int = 10,
        learning_rate: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        使用知识蒸馏训练学生模型
        """
        optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
        
        # 训练历史
        history = {
            'total_loss': [],
            'kd_loss': [],
            'hard_loss': [],
            'expert_loss': [],
            'cache_loss': [],
            'routing_loss': []
        }
        
        batch_size = config['batch_size']
        num_batches = len(train_data) // batch_size
        
        print(f"开始知识蒸馏训练，共{epochs}个epoch，{num_batches}个batch")
        
        for epoch in range(epochs):
            epoch_losses = {key: [] for key in history.keys()}
            
            for batch_idx in range(num_batches):
                # 获取批次数据
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_inputs = train_data[start_idx:end_idx]
                batch_targets = train_targets[start_idx:end_idx]
                
                # 执行蒸馏训练步骤
                loss_info = student_model.distillation_step(
                    input_data=batch_inputs,
                    targets=batch_targets,
                    optimizer=optimizer
                )
                
                # 记录损失
                for key in history.keys():
                    if key in loss_info:
                        epoch_losses[key].append(loss_info[key])
            
            # 计算epoch平均损失
            for key in history.keys():
                if epoch_losses[key]:
                    avg_loss = np.mean(epoch_losses[key])
                    history[key].append(avg_loss)
                else:
                    history[key].append(0.0)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Total Loss: {history['total_loss'][-1]:.4f}")
            print(f"  KD Loss: {history['kd_loss'][-1]:.4f}")
            print(f"  Hard Loss: {history['hard_loss'][-1]:.4f}")
            print(f"  Expert Loss: {history['expert_loss'][-1]:.4f}")
        
        return history
    
    def train_without_distillation(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_targets: torch.Tensor,
        epochs: int = 10,
        learning_rate: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        不使用知识蒸馏的标准训练
        """
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = {'loss': []}
        batch_size = config['batch_size']
        num_batches = len(train_data) // batch_size
        
        print(f"开始标准训练，共{epochs}个epoch，{num_batches}个batch")
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_inputs = train_data[start_idx:end_idx]
                batch_targets = train_targets[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # 前向传播
                if hasattr(model, 'forward') and 'return_loss' in model.forward.__code__.co_varnames:
                    outputs, loss = model(batch_inputs, return_loss=True)
                else:
                    outputs = model(batch_inputs)
                    # 重塑输出和目标以计算损失
                    outputs_2d = outputs.view(-1, outputs.size(-1))
                    targets_1d = batch_targets.view(-1)
                    loss = criterion(outputs_2d, targets_1d)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        return history
    
    def evaluate_model(
        self, 
        model: nn.Module, 
        test_data: torch.Tensor, 
        test_targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        评估模型性能
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        criterion = nn.CrossEntropyLoss()
        batch_size = config['batch_size']
        num_batches = len(test_data) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_inputs = test_data[start_idx:end_idx]
                batch_targets = test_targets[start_idx:end_idx]
                
                # 前向传播
                outputs = model(batch_inputs)
                
                # 计算损失
                outputs_2d = outputs.view(-1, outputs.size(-1))
                targets_1d = batch_targets.view(-1)
                loss = criterion(outputs_2d, targets_1d)
                
                total_loss += loss.item() * batch_inputs.size(0)
                total_samples += batch_inputs.size(0)
                
                # 计算准确率
                predictions = torch.argmax(outputs_2d, dim=1)
                correct_predictions += (predictions == targets_1d).sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / (total_samples * test_data.size(1))  # 考虑序列长度
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': np.exp(avg_loss)
        }
    
    def compare_models(self):
        """
        比较不同模型的性能
        """
        print("=" * 60)
        print("PiKV知识蒸馏性能比较")
        print("=" * 60)
        
        # 生成训练和测试数据
        train_data, train_targets = self.generate_training_data(num_samples=800)
        test_data, test_targets = self.generate_training_data(num_samples=200)
        
        # 创建模型
        print("\n1. 创建模型...")
        
        # 使用知识蒸馏的PiKV模型
        student_distill, teacher_model = self.create_models()
        
        # 不使用知识蒸馏的PiKV模型
        student_no_distill = PiKVMoE(rank=4, alpha=1.0, use_distillation=False).to(self.device)
        
        # 标准MoE模型
        standard_moe = StandardMoE().to(self.device)
        
        # LoRA PiKV模型
        lora_pikv = LoRAPiKVMoE(rank=4, alpha=1.0).to(self.device)
        
        models = {
            'PiKV + Distillation': student_distill,
            'PiKV (No Distillation)': student_no_distill,
            'Standard MoE': standard_moe,
            'LoRA PiKV': lora_pikv
        }
        
        results = {}
        training_histories = {}
        
        # 训练所有模型
        print("\n2. 训练模型...")
        
        for name, model in models.items():
            print(f"\n训练 {name}...")
            start_time = time.time()
            
            if name == 'PiKV + Distillation':
                # 使用知识蒸馏训练
                history = self.train_with_distillation(
                    student_model=model,
                    teacher_model=teacher_model,
                    train_data=train_data,
                    train_targets=train_targets,
                    epochs=5
                )
            else:
                # 标准训练
                history = self.train_without_distillation(
                    model=model,
                    train_data=train_data,
                    train_targets=train_targets,
                    epochs=5
                )
            
            training_time = time.time() - start_time
            training_histories[name] = history
            
            # 评估模型
            print(f"评估 {name}...")
            eval_results = self.evaluate_model(model, test_data, test_targets)
            eval_results['training_time'] = training_time
            
            results[name] = eval_results
            
            print(f"  测试损失: {eval_results['loss']:.4f}")
            print(f"  准确率: {eval_results['accuracy']:.4f}")
            print(f"  困惑度: {eval_results['perplexity']:.4f}")
            print(f"  训练时间: {training_time:.2f}s")
        
        # 保存结果
        self.results = results
        self.training_histories = training_histories
        
        # 绘制结果
        self.plot_comparison_results()
        
        return results
    
    def plot_comparison_results(self):
        """
        绘制比较结果
        """
        if not hasattr(self, 'results') or not hasattr(self, 'training_histories'):
            print("没有可绘制的结果")
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 测试性能比较
        ax1 = axes[0, 0]
        models = list(self.results.keys())
        losses = [self.results[model]['loss'] for model in models]
        accuracies = [self.results[model]['accuracy'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, losses, width, label='Test Loss', alpha=0.8)
        ax1_twin = ax1.twinx()
        ax1_twin.bar(x + width/2, accuracies, width, label='Accuracy', alpha=0.8, color='orange')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Test Loss', color='blue')
        ax1_twin.set_ylabel('Accuracy', color='orange')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. 困惑度比较
        ax2 = axes[0, 1]
        perplexities = [self.results[model]['perplexity'] for model in models]
        bars = ax2.bar(models, perplexities, alpha=0.8, color='green')
        ax2.set_ylabel('Perplexity (lower is better)')
        ax2.set_title('Model Perplexity Comparison')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, perp in zip(bars, perplexities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{perp:.2f}', ha='center', va='bottom')
        
        # 3. 训练时间比较
        ax3 = axes[1, 0]
        training_times = [self.results[model]['training_time'] for model in models]
        bars = ax3.bar(models, training_times, alpha=0.8, color='red')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title('Training Time Comparison')
        ax3.set_xticklabels(models, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, time_val in zip(bars, training_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.1f}s', ha='center', va='bottom')
        
        # 4. 知识蒸馏训练曲线
        ax4 = axes[1, 1]
        if 'PiKV + Distillation' in self.training_histories:
            distill_history = self.training_histories['PiKV + Distillation']
            epochs = range(1, len(distill_history['total_loss']) + 1)
            
            ax4.plot(epochs, distill_history['total_loss'], 'b-', label='Total Loss')
            ax4.plot(epochs, distill_history['kd_loss'], 'r--', label='KD Loss')
            ax4.plot(epochs, distill_history['hard_loss'], 'g--', label='Hard Loss')
            ax4.plot(epochs, distill_history['expert_loss'], 'm--', label='Expert Loss')
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_title('Knowledge Distillation Training Curves')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('distillation_results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"比较结果已保存到: distillation_results/model_comparison.png")
    
    def run_ablation_study(self):
        """
        运行消融实验，测试不同蒸馏参数的效果
        """
        print("\n" + "=" * 60)
        print("知识蒸馏消融实验")
        print("=" * 60)
        
        # 生成数据
        train_data, train_targets = self.generate_training_data(num_samples=400)
        test_data, test_targets = self.generate_training_data(num_samples=100)
        
        # 测试不同的温度参数
        temperatures = [1.0, 2.0, 4.0, 8.0]
        temp_results = {}
        
        for temp in temperatures:
            print(f"\n测试温度参数: {temp}")
            
            # 创建模型
            student_model = PiKVMoE(
                rank=4, 
                alpha=1.0, 
                use_distillation=True,
                teacher_hidden_size=config['hidden_size'] * 2
            ).to(self.device)
            
            # 修改温度参数
            student_model.distillation_module.kd_loss.temperature = temp
            
            teacher_model = create_teacher_model(
                hidden_size=config['hidden_size'] * 2,
                num_experts=config['num_experts']
            ).to(self.device)
            
            # 训练
            history = self.train_with_distillation(
                student_model=student_model,
                teacher_model=teacher_model,
                train_data=train_data,
                train_targets=train_targets,
                epochs=3
            )
            
            # 评估
            eval_results = self.evaluate_model(student_model, test_data, test_targets)
            temp_results[temp] = eval_results
            
            print(f"  最终损失: {eval_results['loss']:.4f}")
            print(f"  准确率: {eval_results['accuracy']:.4f}")
        
        # 绘制温度消融结果
        self.plot_temperature_ablation(temp_results)
        
        return temp_results
    
    def plot_temperature_ablation(self, temp_results: Dict[float, Dict[str, float]]):
        """
        绘制温度消融实验结果
        """
        temperatures = list(temp_results.keys())
        losses = [temp_results[temp]['loss'] for temp in temperatures]
        accuracies = [temp_results[temp]['accuracy'] for temp in temperatures]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 损失曲线
        ax1.plot(temperatures, losses, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Temperature')
        ax1.set_ylabel('Test Loss')
        ax1.set_title('Effect of Temperature on Test Loss')
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(temperatures, accuracies, 'r-o', linewidth=2, markersize=8)
        ax2.set_xlabel('Temperature')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Effect of Temperature on Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('distillation_results/temperature_ablation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"温度消融结果已保存到: distillation_results/temperature_ablation.png")

def main():
    """
    主函数
    """
    print("PiKV知识蒸馏测试")
    print("=" * 60)
    
    # 创建训练器
    trainer = DistillationTrainer()
    
    # 运行模型比较
    results = trainer.compare_models()
    
    # 运行消融实验
    ablation_results = trainer.run_ablation_study()
    
    # 打印总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    
    print("\n模型性能排名（按测试损失）:")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['loss'])
    for i, (model_name, metrics) in enumerate(sorted_models, 1):
        print(f"{i}. {model_name}: Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}")
    
    print("\n最佳温度参数:")
    best_temp = min(ablation_results.items(), key=lambda x: x[1]['loss'])
    print(f"温度 {best_temp[0]}: Loss={best_temp[1]['loss']:.4f}, Accuracy={best_temp[1]['accuracy']:.4f}")
    
    print(f"\n所有结果已保存到 distillation_results/ 目录")

if __name__ == "__main__":
    main() 