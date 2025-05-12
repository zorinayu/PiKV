import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List, Optional, Union

class BaseRouter(nn.Module):
    """
    基础路由器类，为MoE提供路由逻辑
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2, 
        capacity_factor: float = 1.5,
        expert_backend: str = "fairscale"
    ):
        super(BaseRouter, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # 超额容量因子，用于确保每个专家接收到合适数量的令牌
        # 高于1.0表示专家可以处理的token数量超过平均分配量
        self.capacity_factor = capacity_factor
        self.expert_backend = expert_backend
        
        # 计算每个专家的容量
        # 容量 = batch_size * seq_len * capacity_factor * top_k / num_experts
        # 将在forward中动态计算
        
        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
    
    def _compute_capacity(self, batch_size: int, seq_len: int) -> int:
        """计算每个专家的容量"""
        return int(batch_size * seq_len * self.capacity_factor * self.top_k / self.num_experts)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        路由输入到专家
        
        Args:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            expert_mask: 专家可用性掩码 [num_experts]
            
        Returns:
            dispatch_tensor: 调度张量
            combine_tensor: 组合张量
            router_probs: 路由概率
            aux_loss: 辅助损失
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 计算路由逻辑（分数）
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        # 如果提供了专家掩码，应用掩码 
        if expert_mask is not None:
            # 将掩码扩展到匹配路由逻辑的形状
            router_logits = router_logits + (1 - expert_mask.unsqueeze(0).unsqueeze(0)) * -1e9
            
        # 路由概率
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
        
        # 获取top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)  # [batch_size, seq_len, top_k]
        
        # 重新归一化top_k概率
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 计算专家容量
        capacity = self._compute_capacity(batch_size, seq_len)
        
        # 创建调度和组合张量 - 可在不同专家后端实现中使用
        if self.expert_backend == "fairscale":
            # FairScale MoE实现
            # 压平索引和概率
            top_k_indices = top_k_indices.view(-1, self.top_k)  # [batch_size*seq_len, top_k]
            top_k_probs = top_k_probs.view(-1, self.top_k)  # [batch_size*seq_len, top_k]
            
            # 为每个专家创建调度和组合掩码
            dispatch_tensor = torch.zeros(
                batch_size * seq_len, self.num_experts, capacity,
                device=hidden_states.device
            )
            combine_tensor = torch.zeros(
                batch_size * seq_len, self.num_experts, capacity,
                device=hidden_states.device
            )
            
            # 跟踪每个专家当前容量使用情况
            expert_count = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.int32)
            
            # 填充调度和组合掩码
            position_in_batch = torch.arange(batch_size * seq_len, device=hidden_states.device)
            
            for i in range(self.top_k):
                # 当前专家索引
                expert_idx = top_k_indices[:, i]
                # 当前专家概率
                prob = top_k_probs[:, i]
                
                # 检查专家容量
                # mask_capacity标识哪些专家仍有可用容量
                mask_capacity = expert_count[expert_idx] < capacity
                
                # 只分配给有容量的专家
                expert_idx_with_capacity = expert_idx[mask_capacity]
                position_with_capacity = position_in_batch[mask_capacity]
                prob_with_capacity = prob[mask_capacity]
                
                # 更新专家计数并获取每个令牌在专家容量中的位置
                token_positions = expert_count[expert_idx_with_capacity]
                expert_count[expert_idx_with_capacity] += 1
                
                # 更新调度和组合张量
                dispatch_tensor[position_with_capacity, expert_idx_with_capacity, token_positions] = 1.0
                combine_tensor[position_with_capacity, expert_idx_with_capacity, token_positions] = prob_with_capacity
            
            # 重塑调度和组合张量
            dispatch_tensor = dispatch_tensor.view(batch_size, seq_len, self.num_experts, capacity)
            combine_tensor = combine_tensor.view(batch_size, seq_len, self.num_experts, capacity)
            
        else:
            # 基本实现 - 适用于小规模测试
            dispatch_tensor = torch.zeros(
                batch_size, seq_len, self.num_experts, 1,
                device=hidden_states.device
            )
            combine_tensor = torch.zeros(
                batch_size, seq_len, self.num_experts, 1,
                device=hidden_states.device
            )
            
            for i in range(batch_size):
                for j in range(seq_len):
                    for k in range(self.top_k):
                        expert_idx = top_k_indices[i, j, k].item()
                        prob = top_k_probs[i, j, k].item()
                        dispatch_tensor[i, j, expert_idx, 0] = 1.0
                        combine_tensor[i, j, expert_idx, 0] = prob
        
        # 计算辅助损失 - 鼓励负载平衡
        # 1. 专家使用密度
        router_prob_per_expert = router_probs.mean(dim=[0, 1])  # [num_experts]
        
        # 2. 计算负载平衡损失
        # 目标：使所有专家的使用率相同(1/num_experts)
        aux_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss

class TopKBalancedRouter(BaseRouter):
    """
    带负载平衡的TopK路由器
    改进的负载平衡损失，结合专家使用率和专家分配均匀性
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2,
        capacity_factor: float = 1.5,
        expert_backend: str = "fairscale",
        balance_coefficient: float = 0.01,
        balance_mode: str = "entropy"
    ):
        super(TopKBalancedRouter, self).__init__(
            hidden_size, num_experts, top_k, capacity_factor, expert_backend
        )
        self.balance_coefficient = balance_coefficient
        self.balance_mode = balance_mode  # "entropy", "cv" (coefficient of variation)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        dispatch_tensor, combine_tensor, router_probs, aux_loss = super().forward(
            hidden_states, expert_mask
        )
        
        # 进一步改进负载平衡损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])  # [num_experts]
        
        # 更精细的负载均衡损失
        if self.balance_mode == "entropy":
            # 最大化熵 - 使用概率的熵
            balance_term = -torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert + 1e-9))
            # 将其归一化，使最大值为1
            balance_term = balance_term / math.log(self.num_experts)
            # 负值，因为我们想要最大化熵
            balance_loss = -self.balance_coefficient * balance_term
        
        elif self.balance_mode == "cv":
            # 变异系数（标准差/均值）
            # 变异系数越小，分布越均匀
            mean = router_prob_per_expert.mean()
            std = router_prob_per_expert.std()
            cv = std / (mean + 1e-9)
            balance_loss = self.balance_coefficient * cv
        
        else:
            # 默认Z损失 - 最小化第二范数到均匀分布的距离
            uniform = torch.ones_like(router_prob_per_expert) / self.num_experts
            balance_term = torch.sum((router_prob_per_expert - uniform) ** 2)
            balance_loss = self.balance_coefficient * balance_term
        
        # 总损失 = 原损失 + 平衡损失
        aux_loss = aux_loss + balance_loss
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss

class AdaptiveRouter(BaseRouter):
    """
    自适应路由器 - 基于输入特征动态调整路由策略
    参考DeepSeek-V2中的路由方法，综合考虑输入特征重要性
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2,
        capacity_factor: float = 1.5,
        expert_backend: str = "fairscale",
        imp_threshold: float = 0.5
    ):
        super(AdaptiveRouter, self).__init__(
            hidden_size, num_experts, top_k, capacity_factor, expert_backend
        )
        
        # 重要性预测器
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 阈值
        self.imp_threshold = imp_threshold
        
        # 两个路由网络 - 分别针对重要和不重要的token
        self.router_important = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
        self.router_unimportant = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 预测重要性分数
        importance = self.importance_predictor(hidden_states)  # [batch_size, seq_len, 1]
        
        # 创建重要性掩码
        imp_mask = (importance > self.imp_threshold).float()  # [batch_size, seq_len, 1]
        
        # 根据掩码分别路由
        router_logits_imp = self.router_important(hidden_states)
        router_logits_unimp = self.router_unimportant(hidden_states)
        
        # 组合两组路由逻辑
        router_logits = imp_mask * router_logits_imp + (1 - imp_mask) * router_logits_unimp
        
        # 应用专家掩码(如果有)
        if expert_mask is not None:
            router_logits = router_logits + (1 - expert_mask.unsqueeze(0).unsqueeze(0)) * -1e9
            
        # 路由概率
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
        
        # 获取top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)  # [batch_size, seq_len, top_k]
        
        # 归一化top_k概率
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 计算专家容量
        capacity = self._compute_capacity(batch_size, seq_len)
        
        # 创建调度和组合张量
        if self.expert_backend == "fairscale":
            # 与BaseRouter实现类似
            top_k_indices = top_k_indices.view(-1, self.top_k)  # [batch_size*seq_len, top_k]
            top_k_probs = top_k_probs.view(-1, self.top_k)  # [batch_size*seq_len, top_k]
            
            dispatch_tensor = torch.zeros(
                batch_size * seq_len, self.num_experts, capacity,
                device=hidden_states.device
            )
            combine_tensor = torch.zeros(
                batch_size * seq_len, self.num_experts, capacity,
                device=hidden_states.device
            )
            
            expert_count = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.int32)
            position_in_batch = torch.arange(batch_size * seq_len, device=hidden_states.device)
            
            for i in range(self.top_k):
                expert_idx = top_k_indices[:, i]
                prob = top_k_probs[:, i]
                
                mask_capacity = expert_count[expert_idx] < capacity
                
                expert_idx_with_capacity = expert_idx[mask_capacity]
                position_with_capacity = position_in_batch[mask_capacity]
                prob_with_capacity = prob[mask_capacity]
                
                token_positions = expert_count[expert_idx_with_capacity]
                expert_count[expert_idx_with_capacity] += 1
                
                dispatch_tensor[position_with_capacity, expert_idx_with_capacity, token_positions] = 1.0
                combine_tensor[position_with_capacity, expert_idx_with_capacity, token_positions] = prob_with_capacity
            
            dispatch_tensor = dispatch_tensor.view(batch_size, seq_len, self.num_experts, capacity)
            combine_tensor = combine_tensor.view(batch_size, seq_len, self.num_experts, capacity)
            
        else:
            # 简化版实现
            dispatch_tensor = torch.zeros(
                batch_size, seq_len, self.num_experts, 1,
                device=hidden_states.device
            )
            combine_tensor = torch.zeros(
                batch_size, seq_len, self.num_experts, 1,
                device=hidden_states.device
            )
            
            for i in range(batch_size):
                for j in range(seq_len):
                    for k in range(self.top_k):
                        expert_idx = top_k_indices[i, j, k].item()
                        prob = top_k_probs[i, j, k].item()
                        dispatch_tensor[i, j, expert_idx, 0] = 1.0
                        combine_tensor[i, j, expert_idx, 0] = prob
        
        # 改进的负载平衡损失
        # 1. 专家分布均匀性损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])  # [num_experts]
        entropy_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        # 2. 重要性相关损失 - 确保重要的token被更均匀地分配
        # 重要token的路由概率
        imp_probs = router_probs * imp_mask
        imp_probs_sum = imp_probs.sum(dim=[0, 1]) + 1e-9
        imp_probs_per_expert = imp_probs_sum / imp_probs_sum.sum()
        
        # 重要token在各专家之间的分布熵
        imp_entropy = -torch.sum(imp_probs_per_expert * torch.log(imp_probs_per_expert + 1e-9))
        imp_entropy_norm = imp_entropy / math.log(self.num_experts)
        imp_entropy_loss = -0.1 * imp_entropy_norm  # 鼓励重要token均匀分布
        
        # 综合损失
        aux_loss = entropy_loss + imp_entropy_loss
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss, importance

class PiKVRouter(AdaptiveRouter):
    """
    PiKV特定的路由器 - 考虑缓存重要性和历史使用情况
    用于PiKV缓存系统，能够处理关键值缓存的路由
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2,
        capacity_factor: float = 1.5,
        expert_backend: str = "fairscale",
        imp_threshold: float = 0.5,
        cache_update_interval: int = 100
    ):
        super(PiKVRouter, self).__init__(
            hidden_size, num_experts, top_k, capacity_factor, expert_backend, imp_threshold
        )
        
        # 缓存使用历史
        self.register_buffer('cache_usage_history', torch.zeros(num_experts))
        
        # KV缓存特定的预测模型
        self.kv_importance_model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 缓存路由修正因子
        self.cache_correction_factor = nn.Parameter(torch.ones(num_experts))
        
        # 缓存更新计数器和间隔
        self.register_buffer('update_counter', torch.tensor(0))
        self.cache_update_interval = cache_update_interval
    
    def update_cache_usage(self, usage_vector: torch.Tensor):
        """更新缓存使用历史"""
        # 使用指数移动平均
        self.cache_usage_history = 0.9 * self.cache_usage_history + 0.1 * usage_vector
        
        # 更新计数器
        self.update_counter += 1
        
        # 周期性调整修正因子
        if self.update_counter % self.cache_update_interval == 0:
            # 对于使用率较高的缓存，降低其路由概率
            # 注意这里使用了倒数关系
            self.cache_correction_factor.data = 1.0 / (self.cache_usage_history + 0.5)
            # 归一化修正因子
            self.cache_correction_factor.data = self.cache_correction_factor.data / self.cache_correction_factor.data.mean()
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        kv_states: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 标准重要性预测
        importance = self.importance_predictor(hidden_states)  # [batch_size, seq_len, 1]
        
        # 如果提供了KV状态，额外预测KV特定的重要性
        if kv_states is not None:
            kv_importance = self.kv_importance_model(kv_states)
            # 结合两种重要性
            importance = (importance + kv_importance) / 2
        
        # 创建重要性掩码
        imp_mask = (importance > self.imp_threshold).float()
        
        # 分别路由
        router_logits_imp = self.router_important(hidden_states)
        router_logits_unimp = self.router_unimportant(hidden_states)
        
        # 组合路由逻辑
        router_logits = imp_mask * router_logits_imp + (1 - imp_mask) * router_logits_unimp
        
        # 应用缓存修正因子
        router_logits = router_logits * self.cache_correction_factor.unsqueeze(0).unsqueeze(0)
        
        # 应用专家掩码(如果有)
        if expert_mask is not None:
            router_logits = router_logits + (1 - expert_mask.unsqueeze(0).unsqueeze(0)) * -1e9
            
        # 路由概率
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 获取top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        
        # 归一化top_k概率
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 计算专家容量
        capacity = self._compute_capacity(batch_size, seq_len)
        
        # 创建调度和组合张量
        if self.expert_backend == "fairscale":
            top_k_indices = top_k_indices.view(-1, self.top_k)
            top_k_probs = top_k_probs.view(-1, self.top_k)
            
            dispatch_tensor = torch.zeros(
                batch_size * seq_len, self.num_experts, capacity,
                device=hidden_states.device
            )
            combine_tensor = torch.zeros(
                batch_size * seq_len, self.num_experts, capacity,
                device=hidden_states.device
            )
            
            expert_count = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.int32)
            position_in_batch = torch.arange(batch_size * seq_len, device=hidden_states.device)
            
            for i in range(self.top_k):
                expert_idx = top_k_indices[:, i]
                prob = top_k_probs[:, i]
                
                mask_capacity = expert_count[expert_idx] < capacity
                
                expert_idx_with_capacity = expert_idx[mask_capacity]
                position_with_capacity = position_in_batch[mask_capacity]
                prob_with_capacity = prob[mask_capacity]
                
                token_positions = expert_count[expert_idx_with_capacity]
                expert_count[expert_idx_with_capacity] += 1
                
                dispatch_tensor[position_with_capacity, expert_idx_with_capacity, token_positions] = 1.0
                combine_tensor[position_with_capacity, expert_idx_with_capacity, token_positions] = prob_with_capacity
            
            dispatch_tensor = dispatch_tensor.view(batch_size, seq_len, self.num_experts, capacity)
            combine_tensor = combine_tensor.view(batch_size, seq_len, self.num_experts, capacity)
            
        else:
            # 简化实现
            dispatch_tensor = torch.zeros(
                batch_size, seq_len, self.num_experts, 1,
                device=hidden_states.device
            )
            combine_tensor = torch.zeros(
                batch_size, seq_len, self.num_experts, 1,
                device=hidden_states.device
            )
            
            for i in range(batch_size):
                for j in range(seq_len):
                    for k in range(self.top_k):
                        expert_idx = top_k_indices[i, j, k].item()
                        prob = top_k_probs[i, j, k].item()
                        dispatch_tensor[i, j, expert_idx, 0] = 1.0
                        combine_tensor[i, j, expert_idx, 0] = prob
        
        # 使用情况追踪 - 统计路由到每个专家的token数
        expert_usage = router_probs.sum(dim=[0, 1])
        expert_usage_norm = expert_usage / expert_usage.sum()
        
        # 更新缓存使用历史
        self.update_cache_usage(expert_usage_norm)
        
        # 负载平衡损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        uniform = torch.ones_like(router_prob_per_expert) / self.num_experts
        aux_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        # 添加使用历史的修正项
        history_correction = torch.sum(self.cache_usage_history * expert_usage_norm)
        aux_loss = aux_loss + 0.05 * history_correction
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss, importance 