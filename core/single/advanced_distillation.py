"""
Advanced Knowledge Distillation Methods for PiKV
包含最新的蒸馏方法：DistillM, DistillM-2, Speculative KD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class DistillationMethod(Enum):
    """蒸馏方法枚举"""
    DISTILLM = "distillm"
    DISTILLM_2 = "distillm_2"
    SPECULATIVE_KD = "speculative_kd"
    MINILLM = "minillm"
    CLASSIC_KD = "classic_kd"


class DistillMDistillation(nn.Module):
    """
    DistillM: Towards Delicate Imitation of Discriminative Features for Better Knowledge Distillation
    
    核心思想：
    1. 特征对齐：学生模型的特征与教师模型特征对齐
    2. 判别性特征蒸馏：关注最具判别性的特征
    3. 自适应权重：根据特征重要性动态调整蒸馏权重
    """
    
    def __init__(
        self,
        teacher_hidden_size: int,
        student_hidden_size: int,
        num_layers: int = 3,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        feature_alignment_weight: float = 1.0,
        discriminative_weight: float = 0.5
    ):
        super(DistillMDistillation, self).__init__()
        
        self.teacher_hidden_size = teacher_hidden_size
        self.student_hidden_size = student_hidden_size
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.feature_alignment_weight = feature_alignment_weight
        self.discriminative_weight = discriminative_weight
        
        # 特征对齐网络
        self.feature_alignment = nn.ModuleList([
            nn.Sequential(
                nn.Linear(student_hidden_size, teacher_hidden_size),
                nn.LayerNorm(teacher_hidden_size),
                nn.ReLU(),
                nn.Linear(teacher_hidden_size, teacher_hidden_size)
            ) for _ in range(num_layers)
        ])
        
        # 判别性特征提取器
        self.discriminative_extractor = nn.Sequential(
            nn.Linear(teacher_hidden_size, teacher_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(teacher_hidden_size // 2, teacher_hidden_size // 4),
            nn.ReLU(),
            nn.Linear(teacher_hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 自适应权重网络
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(teacher_hidden_size, teacher_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(teacher_hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def compute_feature_alignment_loss(
        self, 
        student_features: List[torch.Tensor], 
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算特征对齐损失"""
        total_loss = 0.0
        
        for i, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
            # 对齐学生特征到教师特征空间
            aligned_student_feat = self.feature_alignment[i](student_feat)
            
            # 计算MSE损失
            alignment_loss = F.mse_loss(aligned_student_feat, teacher_feat)
            total_loss += alignment_loss
        
        return total_loss / len(student_features)
    
    def compute_discriminative_loss(
        self, 
        teacher_features: List[torch.Tensor],
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor
    ) -> torch.Tensor:
        """计算判别性特征蒸馏损失"""
        # 提取判别性特征权重
        discriminative_weights = []
        for teacher_feat in teacher_features:
            weight = self.discriminative_extractor(teacher_feat)
            discriminative_weights.append(weight)
        
        # 计算加权的KL散度损失
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # 使用判别性权重加权KL损失
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none')
        
        # 平均判别性权重
        avg_discriminative_weight = torch.stack(discriminative_weights).mean(dim=0)
        weighted_kl_loss = (kl_loss * avg_discriminative_weight.squeeze(-1)).mean()
        
        return weighted_kl_loss * (self.temperature ** 2)
    
    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播计算蒸馏损失"""
        
        # 特征对齐损失
        feature_loss = self.compute_feature_alignment_loss(student_features, teacher_features)
        
        # 判别性特征蒸馏损失
        discriminative_loss = self.compute_discriminative_loss(
            teacher_features, teacher_logits, student_logits
        )
        
        # 标准KL散度损失
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)
        
        # 总蒸馏损失
        distill_loss = (
            self.feature_alignment_weight * feature_loss +
            self.discriminative_weight * discriminative_loss +
            self.alpha * kl_loss
        )
        
        # 如果有标签，添加交叉熵损失
        total_loss = distill_loss
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            total_loss = self.alpha * distill_loss + self.beta * ce_loss
        
        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss,
            'feature_loss': feature_loss,
            'discriminative_loss': discriminative_loss,
            'kl_loss': kl_loss
        }


class DistillM2Distillation(nn.Module):
    """
    DistillM-2: Enhanced version with multi-scale feature distillation
    
    改进点：
    1. 多尺度特征蒸馏
    2. 注意力引导的特征对齐
    3. 渐进式蒸馏策略
    """
    
    def __init__(
        self,
        teacher_hidden_size: int,
        student_hidden_size: int,
        num_layers: int = 3,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        multi_scale_weights: List[float] = [1.0, 0.8, 0.6],
        attention_weight: float = 0.5,
        progressive_weight: float = 0.3
    ):
        super(DistillM2Distillation, self).__init__()
        
        self.teacher_hidden_size = teacher_hidden_size
        self.student_hidden_size = student_hidden_size
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.multi_scale_weights = multi_scale_weights
        self.attention_weight = attention_weight
        self.progressive_weight = progressive_weight
        
        # 多尺度特征对齐网络
        self.multi_scale_aligners = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(student_hidden_size, teacher_hidden_size),
                    nn.LayerNorm(teacher_hidden_size),
                    nn.GELU(),
                    nn.Linear(teacher_hidden_size, teacher_hidden_size)
                ) for _ in range(num_layers)
            ]) for scale in range(len(multi_scale_weights))
        ])
        
        # 注意力引导网络
        self.attention_guide = nn.MultiheadAttention(
            embed_dim=teacher_hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 渐进式蒸馏控制器
        self.progressive_controller = nn.Sequential(
            nn.Linear(teacher_hidden_size, teacher_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(teacher_hidden_size // 2, num_layers),
            nn.Softmax(dim=-1)
        )
        
        # 特征重要性评估器
        self.importance_estimator = nn.Sequential(
            nn.Linear(teacher_hidden_size, teacher_hidden_size // 4),
            nn.ReLU(),
            nn.Linear(teacher_hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def compute_multi_scale_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算多尺度特征蒸馏损失"""
        total_loss = 0.0
        
        for scale_idx, scale_weight in enumerate(self.multi_scale_weights):
            scale_loss = 0.0
            
            for layer_idx, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
                # 多尺度特征对齐
                aligned_feat = self.multi_scale_aligners[scale_idx][layer_idx](student_feat)
                
                # 计算不同尺度的损失
                if scale_idx == 0:  # 原始尺度
                    loss = F.mse_loss(aligned_feat, teacher_feat)
                elif scale_idx == 1:  # 池化尺度
                    pooled_aligned = F.adaptive_avg_pool1d(
                        aligned_feat.transpose(1, 2), 
                        aligned_feat.size(1) // 2
                    ).transpose(1, 2)
                    pooled_teacher = F.adaptive_avg_pool1d(
                        teacher_feat.transpose(1, 2), 
                        teacher_feat.size(1) // 2
                    ).transpose(1, 2)
                    loss = F.mse_loss(pooled_aligned, pooled_teacher)
                else:  # 更小尺度
                    pooled_aligned = F.adaptive_avg_pool1d(
                        aligned_feat.transpose(1, 2), 
                        aligned_feat.size(1) // 4
                    ).transpose(1, 2)
                    pooled_teacher = F.adaptive_avg_pool1d(
                        teacher_feat.transpose(1, 2), 
                        teacher_feat.size(1) // 4
                    ).transpose(1, 2)
                    loss = F.mse_loss(pooled_aligned, pooled_teacher)
                
                scale_loss += loss
            
            total_loss += scale_weight * scale_loss / len(student_features)
        
        return total_loss
    
    def compute_attention_guided_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算注意力引导的特征对齐损失"""
        total_loss = 0.0
        
        for student_feat, teacher_feat in zip(student_features, teacher_features):
            # 使用教师特征作为query，学生特征作为key和value
            attended_student, _ = self.attention_guide(
                teacher_feat, student_feat, student_feat
            )
            
            # 计算注意力引导的对齐损失
            attention_loss = F.mse_loss(attended_student, teacher_feat)
            total_loss += attention_loss
        
        return total_loss / len(student_features)
    
    def compute_progressive_loss(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算渐进式蒸馏损失"""
        # 计算层级重要性权重
        avg_teacher_feat = torch.stack(teacher_features).mean(dim=0)
        layer_weights = self.progressive_controller(avg_teacher_feat.mean(dim=1))
        
        total_loss = 0.0
        for i, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
            # 特征重要性
            importance = self.importance_estimator(teacher_feat)
            
            # 加权损失
            layer_weight = layer_weights[:, i].unsqueeze(-1).unsqueeze(-1)
            weighted_loss = F.mse_loss(student_feat, teacher_feat, reduction='none')
            weighted_loss = (weighted_loss * importance * layer_weight).mean()
            
            total_loss += weighted_loss
        
        return total_loss / len(student_features)
    
    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播计算蒸馏损失"""
        
        # 多尺度特征损失
        multi_scale_loss = self.compute_multi_scale_loss(student_features, teacher_features)
        
        # 注意力引导损失
        attention_loss = self.compute_attention_guided_loss(student_features, teacher_features)
        
        # 渐进式蒸馏损失
        progressive_loss = self.compute_progressive_loss(student_features, teacher_features)
        
        # 标准KL散度损失
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)
        
        # 总蒸馏损失
        distill_loss = (
            multi_scale_loss +
            self.attention_weight * attention_loss +
            self.progressive_weight * progressive_loss +
            self.alpha * kl_loss
        )
        
        # 如果有标签，添加交叉熵损失
        total_loss = distill_loss
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            total_loss = self.alpha * distill_loss + self.beta * ce_loss
        
        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss,
            'multi_scale_loss': multi_scale_loss,
            'attention_loss': attention_loss,
            'progressive_loss': progressive_loss,
            'kl_loss': kl_loss
        }


class SpeculativeKDDistillation(nn.Module):
    """
    Speculative Knowledge Distillation
    基于Google Research的投机性知识蒸馏
    
    核心思想：
    1. 投机性预测：学生模型预测教师模型的行为
    2. 验证机制：验证预测的准确性
    3. 自适应调整：根据预测准确性调整蒸馏策略
    """
    
    def __init__(
        self,
        teacher_hidden_size: int,
        student_hidden_size: int,
        num_speculation_steps: int = 3,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        speculation_weight: float = 0.4,
        verification_weight: float = 0.3,
        adaptive_threshold: float = 0.8
    ):
        super(SpeculativeKDDistillation, self).__init__()
        
        self.teacher_hidden_size = teacher_hidden_size
        self.student_hidden_size = student_hidden_size
        self.num_speculation_steps = num_speculation_steps
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.speculation_weight = speculation_weight
        self.verification_weight = verification_weight
        self.adaptive_threshold = adaptive_threshold
        
        # 投机性预测网络
        self.speculation_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(student_hidden_size, teacher_hidden_size),
                nn.LayerNorm(teacher_hidden_size),
                nn.GELU(),
                nn.Linear(teacher_hidden_size, teacher_hidden_size),
                nn.LayerNorm(teacher_hidden_size)
            ) for _ in range(num_speculation_steps)
        ])
        
        # 验证网络
        self.verification_net = nn.Sequential(
            nn.Linear(teacher_hidden_size * 2, teacher_hidden_size),
            nn.ReLU(),
            nn.Linear(teacher_hidden_size, teacher_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(teacher_hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 自适应权重控制器
        self.adaptive_controller = nn.Sequential(
            nn.Linear(teacher_hidden_size, teacher_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(teacher_hidden_size // 2, num_speculation_steps),
            nn.Softmax(dim=-1)
        )
        
        # 预测准确性跟踪
        self.register_buffer('prediction_accuracy', torch.zeros(num_speculation_steps))
        self.register_buffer('update_count', torch.zeros(1))
    
    def speculative_prediction(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """执行投机性预测"""
        predictions = []
        verification_scores = []
        
        for step in range(self.num_speculation_steps):
            step_predictions = []
            step_verifications = []
            
            for i, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
                # 投机性预测教师特征
                predicted_teacher = self.speculation_predictor[step](student_feat)
                step_predictions.append(predicted_teacher)
                
                # 验证预测准确性
                concat_feat = torch.cat([predicted_teacher, teacher_feat], dim=-1)
                verification_score = self.verification_net(concat_feat)
                step_verifications.append(verification_score)
            
            predictions.append(step_predictions)
            verification_scores.append(step_verifications)
        
        return predictions, verification_scores
    
    def compute_speculation_loss(
        self,
        predictions: List[List[torch.Tensor]],
        teacher_features: List[torch.Tensor],
        verification_scores: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        """计算投机性预测损失"""
        total_loss = 0.0
        
        for step, (step_predictions, step_verifications) in enumerate(zip(predictions, verification_scores)):
            step_loss = 0.0
            
            for pred, teacher_feat, verification in zip(step_predictions, teacher_features, step_verifications):
                # 预测损失
                pred_loss = F.mse_loss(pred, teacher_feat, reduction='none')
                
                # 使用验证分数加权
                weighted_pred_loss = (pred_loss * verification).mean()
                step_loss += weighted_pred_loss
            
            # 计算预测准确性
            avg_verification = torch.stack(step_verifications).mean()
            accuracy = (avg_verification > self.adaptive_threshold).float().mean()
            
            # 更新准确性跟踪
            self.prediction_accuracy[step] = (
                0.9 * self.prediction_accuracy[step] + 0.1 * accuracy
            )
            
            total_loss += step_loss / len(step_predictions)
        
        return total_loss / self.num_speculation_steps
    
    def compute_verification_loss(
        self,
        verification_scores: List[List[torch.Tensor]],
        predictions: List[List[torch.Tensor]],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算验证损失"""
        total_loss = 0.0
        
        for step_verifications, step_predictions in zip(verification_scores, predictions):
            step_loss = 0.0
            
            for verification, pred, teacher_feat in zip(step_verifications, step_predictions, teacher_features):
                # 计算真实的预测准确性
                pred_error = F.mse_loss(pred, teacher_feat, reduction='none').mean(dim=-1, keepdim=True)
                true_accuracy = torch.exp(-pred_error)  # 转换为0-1范围的准确性
                
                # 验证损失
                verification_loss = F.mse_loss(verification, true_accuracy)
                step_loss += verification_loss
            
            total_loss += step_loss / len(step_verifications)
        
        return total_loss / self.num_speculation_steps
    
    def compute_adaptive_weights(self, teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """计算自适应权重"""
        avg_teacher_feat = torch.stack(teacher_features).mean(dim=0).mean(dim=1)
        adaptive_weights = self.adaptive_controller(avg_teacher_feat)
        
        # 结合预测准确性调整权重
        accuracy_weights = self.prediction_accuracy.unsqueeze(0).expand(adaptive_weights.size(0), -1)
        final_weights = adaptive_weights * accuracy_weights
        final_weights = final_weights / final_weights.sum(dim=-1, keepdim=True)
        
        return final_weights
    
    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播计算蒸馏损失"""
        
        # 投机性预测
        predictions, verification_scores = self.speculative_prediction(
            student_features, teacher_features
        )
        
        # 投机性预测损失
        speculation_loss = self.compute_speculation_loss(
            predictions, teacher_features, verification_scores
        )
        
        # 验证损失
        verification_loss = self.compute_verification_loss(
            verification_scores, predictions, teacher_features
        )
        
        # 标准KL散度损失
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kl_loss = kl_loss * (self.temperature ** 2)
        
        # 自适应权重
        adaptive_weights = self.compute_adaptive_weights(teacher_features)
        weighted_speculation_loss = (speculation_loss * adaptive_weights.mean()).sum()
        
        # 总蒸馏损失
        distill_loss = (
            self.speculation_weight * weighted_speculation_loss +
            self.verification_weight * verification_loss +
            self.alpha * kl_loss
        )
        
        # 如果有标签，添加交叉熵损失
        total_loss = distill_loss
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits, labels)
            total_loss = self.alpha * distill_loss + self.beta * ce_loss
        
        # 更新计数
        self.update_count += 1
        
        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss,
            'speculation_loss': speculation_loss,
            'verification_loss': verification_loss,
            'kl_loss': kl_loss,
            'prediction_accuracy': self.prediction_accuracy.mean()
        }


class AdvancedDistillationManager:
    """
    高级蒸馏管理器，统一管理所有蒸馏方法
    """
    
    def __init__(
        self,
        teacher_hidden_size: int,
        student_hidden_size: int,
        method: DistillationMethod = DistillationMethod.DISTILLM,
        **kwargs
    ):
        self.teacher_hidden_size = teacher_hidden_size
        self.student_hidden_size = student_hidden_size
        self.method = method
        
        # 根据方法创建相应的蒸馏器
        if method == DistillationMethod.DISTILLM:
            self.distiller = DistillMDistillation(
                teacher_hidden_size, student_hidden_size, **kwargs
            )
        elif method == DistillationMethod.DISTILLM_2:
            self.distiller = DistillM2Distillation(
                teacher_hidden_size, student_hidden_size, **kwargs
            )
        elif method == DistillationMethod.SPECULATIVE_KD:
            self.distiller = SpeculativeKDDistillation(
                teacher_hidden_size, student_hidden_size, **kwargs
            )
        else:
            raise ValueError(f"Unsupported distillation method: {method}")
    
    def distill(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """执行蒸馏"""
        return self.distiller(
            student_features, teacher_features, student_logits, teacher_logits, labels
        )
    
    def get_method_info(self) -> Dict[str, str]:
        """获取方法信息"""
        method_info = {
            DistillationMethod.DISTILLM: {
                "name": "DistillM",
                "description": "Discriminative feature distillation with adaptive weighting",
                "paper": "DistillM: Towards Delicate Imitation of Discriminative Features"
            },
            DistillationMethod.DISTILLM_2: {
                "name": "DistillM-2",
                "description": "Enhanced multi-scale feature distillation with attention guidance",
                "paper": "DistillM-2: Enhanced Knowledge Distillation"
            },
            DistillationMethod.SPECULATIVE_KD: {
                "name": "Speculative KD",
                "description": "Speculative knowledge distillation with verification mechanism",
                "paper": "Speculative Knowledge Distillation (Google Research)"
            }
        }
        
        return method_info.get(self.method, {"name": "Unknown", "description": "Unknown method"}) 