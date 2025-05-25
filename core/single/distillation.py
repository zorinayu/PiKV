import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List, Optional, Union, Any
from .config import config

class KnowledgeDistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    实现多种蒸馏策略：标准KD、特征匹配、注意力转移
    """
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        feature_weight: float = 0.5,
        attention_weight: float = 0.3
    ):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重
        self.beta = beta    # 真实标签损失权重
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        student_attention: Optional[torch.Tensor] = None,
        teacher_attention: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算知识蒸馏损失
        
        Args:
            student_logits: 学生模型输出 [batch_size, seq_len, vocab_size]
            teacher_logits: 教师模型输出 [batch_size, seq_len, vocab_size]
            student_features: 学生模型特征 [batch_size, seq_len, hidden_size]
            teacher_features: 教师模型特征 [batch_size, seq_len, hidden_size]
            student_attention: 学生模型注意力权重
            teacher_attention: 教师模型注意力权重
            targets: 真实标签 [batch_size, seq_len]
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细信息
        """
        loss_dict = {}
        
        # 1. 标准知识蒸馏损失 (Soft Target Loss)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kd_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        loss_dict['kd_loss'] = kd_loss
        
        # 2. 真实标签损失 (Hard Target Loss)
        hard_loss = torch.tensor(0.0, device=student_logits.device)
        if targets is not None:
            # 重塑为2D进行交叉熵计算
            student_logits_2d = student_logits.view(-1, student_logits.size(-1))
            targets_1d = targets.view(-1)
            hard_loss = self.ce_loss(student_logits_2d, targets_1d)
            loss_dict['hard_loss'] = hard_loss
        
        # 3. 特征匹配损失 (Feature Matching Loss)
        feature_loss = torch.tensor(0.0, device=student_logits.device)
        if student_features is not None and teacher_features is not None:
            # 确保特征维度匹配
            if student_features.shape != teacher_features.shape:
                # 如果维度不匹配，使用线性投影对齐
                if not hasattr(self, 'feature_adapter'):
                    self.feature_adapter = nn.Linear(
                        student_features.size(-1), 
                        teacher_features.size(-1)
                    ).to(student_features.device)
                student_features = self.feature_adapter(student_features)
            
            feature_loss = self.mse_loss(student_features, teacher_features)
            loss_dict['feature_loss'] = feature_loss
        
        # 4. 注意力转移损失 (Attention Transfer Loss)
        attention_loss = torch.tensor(0.0, device=student_logits.device)
        if student_attention is not None and teacher_attention is not None:
            # 计算注意力图的MSE损失
            attention_loss = self.mse_loss(student_attention, teacher_attention)
            loss_dict['attention_loss'] = attention_loss
        
        # 5. 总损失计算
        total_loss = (
            self.alpha * kd_loss + 
            self.beta * hard_loss + 
            self.feature_weight * feature_loss + 
            self.attention_weight * attention_loss
        )
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict

class DistillationLayer(nn.Module):
    """
    蒸馏层 - 用于特征对齐和知识转移
    """
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        hidden_dim: Optional[int] = None,
        use_attention: bool = True
    ):
        super(DistillationLayer, self).__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.hidden_dim = hidden_dim or min(student_dim, teacher_dim)
        self.use_attention = use_attention
        
        # 特征对齐网络
        self.feature_adapter = nn.Sequential(
            nn.Linear(student_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, teacher_dim)
        )
        
        # 注意力对齐网络
        if use_attention:
            self.attention_adapter = nn.Sequential(
                nn.Linear(student_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, teacher_dim),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        对齐学生和教师特征
        
        Args:
            student_features: 学生特征 [batch_size, seq_len, student_dim]
            teacher_features: 教师特征 [batch_size, seq_len, teacher_dim]
        
        Returns:
            aligned_student: 对齐后的学生特征
            attention_weights: 注意力权重（可选）
        """
        # 特征对齐
        aligned_student = self.feature_adapter(student_features)
        
        # 注意力对齐
        attention_weights = None
        if self.use_attention:
            attention_weights = self.attention_adapter(student_features)
            aligned_student = aligned_student * attention_weights
        
        return aligned_student, attention_weights

class PiKVDistillation(nn.Module):
    """
    PiKV专用的知识蒸馏模块
    支持MoE专家级别的蒸馏和KV缓存蒸馏
    """
    def __init__(
        self,
        student_hidden_size: int,
        teacher_hidden_size: int,
        num_experts: int,
        temperature: float = 4.0,
        expert_distill_weight: float = 0.4,
        cache_distill_weight: float = 0.3
    ):
        super(PiKVDistillation, self).__init__()
        self.student_hidden_size = student_hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        self.num_experts = num_experts
        self.expert_distill_weight = expert_distill_weight
        self.cache_distill_weight = cache_distill_weight
        
        # 主要蒸馏损失
        self.kd_loss = KnowledgeDistillationLoss(temperature=temperature)
        
        # 专家级别的蒸馏层
        self.expert_distill_layers = nn.ModuleList([
            DistillationLayer(student_hidden_size, teacher_hidden_size)
            for _ in range(num_experts)
        ])
        
        # KV缓存蒸馏层
        self.cache_distill_layer = DistillationLayer(
            student_hidden_size, 
            teacher_hidden_size,
            use_attention=True
        )
        
        # 路由蒸馏 - 对齐专家选择策略
        self.routing_distill = nn.Sequential(
            nn.Linear(student_hidden_size, teacher_hidden_size),
            nn.ReLU(),
            nn.Linear(teacher_hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def distill_expert_outputs(
        self,
        student_expert_outputs: List[torch.Tensor],
        teacher_expert_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        蒸馏专家输出
        
        Args:
            student_expert_outputs: 学生专家输出列表
            teacher_expert_outputs: 教师专家输出列表
        
        Returns:
            expert_distill_loss: 专家蒸馏损失
        """
        expert_losses = []
        
        for i, (student_out, teacher_out) in enumerate(
            zip(student_expert_outputs, teacher_expert_outputs)
        ):
            # 对齐专家特征
            aligned_student, _ = self.expert_distill_layers[i](student_out, teacher_out)
            
            # 计算MSE损失
            expert_loss = F.mse_loss(aligned_student, teacher_out)
            expert_losses.append(expert_loss)
        
        return torch.stack(expert_losses).mean()
    
    def distill_kv_cache(
        self,
        student_cache: torch.Tensor,
        teacher_cache: torch.Tensor
    ) -> torch.Tensor:
        """
        蒸馏KV缓存
        
        Args:
            student_cache: 学生KV缓存
            teacher_cache: 教师KV缓存
        
        Returns:
            cache_distill_loss: 缓存蒸馏损失
        """
        # 对齐缓存特征
        aligned_cache, attention = self.cache_distill_layer(student_cache, teacher_cache)
        
        # 计算缓存蒸馏损失
        cache_loss = F.mse_loss(aligned_cache, teacher_cache)
        
        # 添加注意力正则化
        if attention is not None:
            attention_reg = torch.mean(attention * torch.log(attention + 1e-8))
            cache_loss = cache_loss - 0.1 * attention_reg
        
        return cache_loss
    
    def distill_routing(
        self,
        student_features: torch.Tensor,
        teacher_routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        蒸馏路由策略
        
        Args:
            student_features: 学生特征
            teacher_routing_weights: 教师路由权重
        
        Returns:
            routing_distill_loss: 路由蒸馏损失
        """
        # 预测学生的路由权重
        student_routing_pred = self.routing_distill(student_features)
        
        # 计算KL散度损失
        routing_loss = F.kl_div(
            F.log_softmax(student_routing_pred, dim=-1),
            F.softmax(teacher_routing_weights, dim=-1),
            reduction='batchmean'
        )
        
        return routing_loss
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_expert_outputs: Optional[List[torch.Tensor]] = None,
        teacher_expert_outputs: Optional[List[torch.Tensor]] = None,
        student_cache: Optional[torch.Tensor] = None,
        teacher_cache: Optional[torch.Tensor] = None,
        teacher_routing_weights: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        完整的PiKV蒸馏前向传播
        
        Returns:
            total_loss: 总蒸馏损失
            loss_dict: 详细损失信息
        """
        # 1. 基础知识蒸馏
        base_loss, loss_dict = self.kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_features=student_features,
            teacher_features=teacher_features,
            targets=targets
        )
        
        # 2. 专家蒸馏
        expert_loss = torch.tensor(0.0, device=student_logits.device)
        if student_expert_outputs and teacher_expert_outputs:
            expert_loss = self.distill_expert_outputs(
                student_expert_outputs, teacher_expert_outputs
            )
            loss_dict['expert_loss'] = expert_loss
        
        # 3. KV缓存蒸馏
        cache_loss = torch.tensor(0.0, device=student_logits.device)
        if student_cache is not None and teacher_cache is not None:
            cache_loss = self.distill_kv_cache(student_cache, teacher_cache)
            loss_dict['cache_loss'] = cache_loss
        
        # 4. 路由蒸馏
        routing_loss = torch.tensor(0.0, device=student_logits.device)
        if teacher_routing_weights is not None:
            routing_loss = self.distill_routing(student_features, teacher_routing_weights)
            loss_dict['routing_loss'] = routing_loss
        
        # 5. 总损失
        total_loss = (
            base_loss + 
            self.expert_distill_weight * expert_loss + 
            self.cache_distill_weight * cache_loss + 
            0.2 * routing_loss
        )
        
        loss_dict['total_distill_loss'] = total_loss
        
        return total_loss, loss_dict

def create_teacher_model(
    hidden_size: int,
    num_experts: int,
    num_layers: int = 6
) -> nn.Module:
    """
    创建教师模型 - 通常是更大、更强的模型
    
    Args:
        hidden_size: 隐藏层大小
        num_experts: 专家数量
        num_layers: 层数
    
    Returns:
        teacher_model: 教师模型
    """
    class TeacherMoE(nn.Module):
        def __init__(self):
            super(TeacherMoE, self).__init__()
            self.hidden_size = hidden_size
            self.num_experts = num_experts
            
            # 更大的专家网络
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU()
                ) for _ in range(num_experts)
            ])
            
            # 路由网络
            self.router = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_experts),
                nn.Softmax(dim=-1)
            )
            
            # 输出投影
            self.output_proj = nn.Linear(hidden_size, config.get('vocab_size', 50257))
        
        def forward(self, x):
            batch_size, seq_len, hidden_size = x.shape
            
            # 路由权重
            routing_weights = self.router(x)
            
            # 专家输出
            expert_outputs = []
            combined_output = torch.zeros_like(x)
            
            for i, expert in enumerate(self.experts):
                expert_out = expert(x)
                expert_outputs.append(expert_out)
                combined_output += expert_out * routing_weights[:, :, i:i+1]
            
            # 输出logits
            logits = self.output_proj(combined_output)
            
            return {
                'logits': logits,
                'features': combined_output,
                'expert_outputs': expert_outputs,
                'routing_weights': routing_weights
            }
    
    return TeacherMoE()

def distillation_training_step(
    student_model: nn.Module,
    teacher_model: nn.Module,
    distillation_module: PiKVDistillation,
    input_data: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, float]:
    """
    单步蒸馏训练
    
    Args:
        student_model: 学生模型
        teacher_model: 教师模型
        distillation_module: 蒸馏模块
        input_data: 输入数据
        targets: 目标标签
        optimizer: 优化器
    
    Returns:
        loss_info: 损失信息字典
    """
    # 教师模型推理（不计算梯度）
    teacher_model.eval()
    with torch.no_grad():
        teacher_outputs = teacher_model(input_data)
    
    # 学生模型推理
    student_model.train()
    student_outputs = student_model(input_data)
    
    # 计算蒸馏损失
    distill_loss, loss_dict = distillation_module(
        student_logits=student_outputs.get('logits', student_outputs),
        teacher_logits=teacher_outputs.get('logits', teacher_outputs),
        student_features=student_outputs.get('features'),
        teacher_features=teacher_outputs.get('features'),
        student_expert_outputs=student_outputs.get('expert_outputs'),
        teacher_expert_outputs=teacher_outputs.get('expert_outputs'),
        teacher_routing_weights=teacher_outputs.get('routing_weights'),
        targets=targets
    )
    
    # 反向传播
    if optimizer is not None:
        optimizer.zero_grad()
        distill_loss.backward()
        optimizer.step()
    
    # 转换损失为标量值
    loss_info = {}
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            loss_info[key] = value.item()
        else:
            loss_info[key] = value
    
    return loss_info 