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

class EPLBRouter(BaseRouter):
    """
    Expert-level Load Balancing (EPLB) Router
    基于DeepSeek EPLB论文的实现，提供更精细的专家级负载平衡
    
    Reference: https://github.com/deepseek-ai/EPLB
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2,
        capacity_factor: float = 1.5,
        expert_backend: str = "fairscale",
        balance_coefficient: float = 0.01,
        temperature: float = 1.0,
        use_auxiliary_loss: bool = True,
        use_z_loss: bool = True,
        z_loss_coefficient: float = 1e-3
    ):
        super(EPLBRouter, self).__init__(
            hidden_size, num_experts, top_k, capacity_factor, expert_backend
        )
        self.balance_coefficient = balance_coefficient
        self.temperature = temperature
        self.use_auxiliary_loss = use_auxiliary_loss
        self.use_z_loss = use_z_loss
        self.z_loss_coefficient = z_loss_coefficient
        
        # EPLB特有的组件
        self.expert_load_tracker = torch.zeros(num_experts)
        self.global_step = 0
        
        # 改进的路由网络，使用更深的网络
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_experts)
        )
        
        # 专家负载平衡的动态权重
        self.register_buffer('expert_weights', torch.ones(num_experts))
        
    def _compute_load_balancing_loss(self, router_probs: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        计算EPLB的负载平衡损失
        """
        batch_size, seq_len, _ = router_probs.shape
        
        # 计算每个专家的使用频率
        expert_usage = torch.zeros(self.num_experts, device=router_probs.device)
        for i in range(self.num_experts):
            expert_usage[i] = (expert_indices == i).float().sum()
        
        # 归一化使用频率
        expert_usage = expert_usage / (batch_size * seq_len * self.top_k)
        
        # 计算理想的均匀分布
        uniform_prob = 1.0 / self.num_experts
        
        # 使用KL散度作为负载平衡损失
        kl_loss = F.kl_div(
            torch.log(expert_usage + 1e-8),
            torch.full_like(expert_usage, uniform_prob),
            reduction='sum'
        )
        
        return kl_loss
    
    def _compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        计算Z损失，防止路由器输出过大的logits
        """
        return torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
    
    def _update_expert_weights(self, expert_usage: torch.Tensor):
        """
        动态更新专家权重以实现更好的负载平衡
        """
        # 计算专家使用率的倒数作为权重
        avg_usage = expert_usage.mean()
        self.expert_weights = avg_usage / (expert_usage + 1e-8)
        
        # 平滑更新
        momentum = 0.9
        self.expert_weights = momentum * self.expert_weights + (1 - momentum) * self.expert_weights
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 计算路由logits
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        # 应用温度缩放
        router_logits = router_logits / self.temperature
        
        # 应用专家权重进行负载平衡
        router_logits = router_logits * self.expert_weights.unsqueeze(0).unsqueeze(0)
        
        # 应用专家掩码
        if expert_mask is not None:
            router_logits = router_logits + (1 - expert_mask.unsqueeze(0).unsqueeze(0)) * -1e9
        
        # 计算路由概率
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 获取top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        
        # 重新归一化
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 计算容量
        capacity = self._compute_capacity(batch_size, seq_len)
        
        # 创建调度和组合张量（使用基类的逻辑）
        dispatch_tensor = torch.zeros(
            batch_size * seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        combine_tensor = torch.zeros(
            batch_size * seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        
        # 专家计数器
        expert_count = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.int32)
        position_in_batch = torch.arange(batch_size * seq_len, device=hidden_states.device)
        
        # 压平索引和概率
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        top_k_probs_flat = top_k_probs.view(-1, self.top_k)
        
        # 填充调度和组合张量
        for i in range(self.top_k):
            expert_idx = top_k_indices_flat[:, i]
            prob = top_k_probs_flat[:, i]
            
            # 检查容量
            mask_capacity = expert_count[expert_idx] < capacity
            expert_idx_with_capacity = expert_idx[mask_capacity]
            position_with_capacity = position_in_batch[mask_capacity]
            prob_with_capacity = prob[mask_capacity]
            
            # 更新计数和张量
            token_positions = expert_count[expert_idx_with_capacity]
            expert_count[expert_idx_with_capacity] += 1
            
            dispatch_tensor[position_with_capacity, expert_idx_with_capacity, token_positions] = 1.0
            combine_tensor[position_with_capacity, expert_idx_with_capacity, token_positions] = prob_with_capacity
        
        # 重塑张量
        dispatch_tensor = dispatch_tensor.view(batch_size, seq_len, self.num_experts, capacity)
        combine_tensor = combine_tensor.view(batch_size, seq_len, self.num_experts, capacity)
        
        # 计算损失
        aux_loss = 0.0
        
        if self.use_auxiliary_loss:
            # 负载平衡损失
            balance_loss = self._compute_load_balancing_loss(router_probs, top_k_indices)
            aux_loss += self.balance_coefficient * balance_loss
        
        if self.use_z_loss:
            # Z损失
            z_loss = self._compute_z_loss(router_logits)
            aux_loss += self.z_loss_coefficient * z_loss
        
        # 更新专家使用统计
        expert_usage = torch.zeros(self.num_experts, device=hidden_states.device)
        for i in range(self.num_experts):
            expert_usage[i] = (top_k_indices == i).float().sum()
        
        self._update_expert_weights(expert_usage)
        self.global_step += 1
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss

class HierarchicalRouter(BaseRouter):
    """
    分层路由器 - 先选择专家组，再选择组内专家
    适用于大规模专家系统
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2,
        capacity_factor: float = 1.5,
        expert_backend: str = "fairscale",
        num_groups: int = 4,
        group_top_k: int = 1
    ):
        super(HierarchicalRouter, self).__init__(
            hidden_size, num_experts, top_k, capacity_factor, expert_backend
        )
        
        self.num_groups = num_groups
        self.group_top_k = group_top_k
        self.experts_per_group = num_experts // num_groups
        
        # 组级路由器
        self.group_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_groups)
        )
        
        # 专家级路由器（每组一个）
        self.expert_routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, self.experts_per_group)
            ) for _ in range(num_groups)
        ])
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 第一阶段：选择专家组
        group_logits = self.group_router(hidden_states)  # [batch_size, seq_len, num_groups]
        group_probs = F.softmax(group_logits, dim=-1)
        
        # 选择top_k组
        top_k_group_probs, top_k_group_indices = torch.topk(
            group_probs, k=min(self.group_top_k, self.num_groups), dim=-1
        )
        
        # 第二阶段：在选中的组内选择专家
        all_expert_probs = []
        all_expert_indices = []
        
        for group_idx in range(self.num_groups):
            # 计算该组的专家logits
            expert_logits = self.expert_routers[group_idx](hidden_states)
            expert_probs = F.softmax(expert_logits, dim=-1)
            
            # 转换为全局专家索引
            global_expert_indices = torch.arange(
                group_idx * self.experts_per_group,
                (group_idx + 1) * self.experts_per_group,
                device=hidden_states.device
            )
            
            all_expert_probs.append(expert_probs)
            all_expert_indices.append(global_expert_indices)
        
        # 组合所有专家的概率
        router_probs = torch.zeros(batch_size, seq_len, self.num_experts, device=hidden_states.device)
        
        for group_idx in range(self.num_groups):
            group_weight = group_probs[:, :, group_idx:group_idx+1]  # [batch_size, seq_len, 1]
            expert_probs = all_expert_probs[group_idx]  # [batch_size, seq_len, experts_per_group]
            
            start_idx = group_idx * self.experts_per_group
            end_idx = (group_idx + 1) * self.experts_per_group
            
            router_probs[:, :, start_idx:end_idx] = group_weight * expert_probs
        
        # 应用专家掩码
        if expert_mask is not None:
            router_probs = router_probs * expert_mask.unsqueeze(0).unsqueeze(0)
        
        # 获取最终的top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 创建调度和组合张量（使用基类逻辑）
        capacity = self._compute_capacity(batch_size, seq_len)
        
        dispatch_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        combine_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        
        # 简化的分配逻辑
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(self.top_k):
                    expert_idx = top_k_indices[i, j, k].item()
                    prob = top_k_probs[i, j, k].item()
                    dispatch_tensor[i, j, expert_idx, 0] = 1.0
                    combine_tensor[i, j, expert_idx, 0] = prob
        
        # 计算辅助损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss


# ============================================================================
# MoE策略集成
# ============================================================================

class FlexMoERouter(BaseRouter):
    """
    Flex-MoE路由器：用于处理任意模态组合的多模态学习场景
    参考：https://github.com/UNITES-Lab/Flex-MoE
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 4,
        modality_dict: Optional[Dict[str, int]] = None,
        use_generalized_router: bool = True
    ):
        super().__init__(hidden_size, num_experts, top_k)
        
        # 模态字典，定义不同模态的映射
        self.modality_dict = modality_dict or {
            'image': 0,
            'genomic': 1, 
            'clinical': 2,
            'biospecimen': 3
        }
        self.num_modalities = len(self.modality_dict)
        
        # 广义路由器 (G-Router)
        self.generalized_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        ) if use_generalized_router else None
        
        # 专门路由器 (S-Router) - 为每种模态组合创建专门的路由器
        self.specialized_routers = nn.ModuleDict()
        for modality_name in self.modality_dict.keys():
            self.specialized_routers[modality_name] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_experts)
            )
        
        # 缺失模态处理
        self.missing_modality_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        modality_info: Optional[Dict[str, torch.Tensor]] = None,
        expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Flex-MoE路由逻辑
        
        Args:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            modality_info: 模态信息字典，包含可用的模态数据
            expert_mask: 专家可用性掩码
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if modality_info is None:
            # 如果没有模态信息，使用标准路由
            return super().forward(hidden_states, expert_mask)
        
        # 处理缺失模态
        available_modalities = list(modality_info.keys())
        missing_modalities = [mod for mod in self.modality_dict.keys() 
                            if mod not in available_modalities]
        
        # 为缺失模态生成表示
        if missing_modalities:
            missing_representation = self.missing_modality_encoder(hidden_states)
            for missing_mod in missing_modalities:
                modality_info[missing_mod] = missing_representation
        
        # 使用广义路由器处理完整模态组合
        if self.generalized_router is not None:
            g_router_logits = self.generalized_router(hidden_states)
            g_router_probs = F.softmax(g_router_logits, dim=-1)
        else:
            g_router_probs = torch.zeros(batch_size, seq_len, self.num_experts, 
                                       device=hidden_states.device)
        
        # 使用专门路由器处理当前模态组合
        s_router_probs = torch.zeros(batch_size, seq_len, self.num_experts, 
                                   device=hidden_states.device)
        
        for modality_name, modality_data in modality_info.items():
            if modality_name in self.specialized_routers:
                modality_logits = self.specialized_routers[modality_name](modality_data)
                modality_probs = F.softmax(modality_logits, dim=-1)
                s_router_probs += modality_probs
        
        # 归一化专门路由器概率
        if len(available_modalities) > 0:
            s_router_probs = s_router_probs / len(available_modalities)
        
        # 组合广义和专门路由器
        alpha = 0.7  # 可调节的权重
        router_probs = alpha * g_router_probs + (1 - alpha) * s_router_probs
        
        # 应用专家掩码
        if expert_mask is not None:
            router_probs = router_probs * expert_mask.unsqueeze(0).unsqueeze(0)
        
        # 获取top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 创建调度和组合张量
        capacity = self._compute_capacity(batch_size, seq_len)
        
        dispatch_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        combine_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        
        # 分配逻辑
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(self.top_k):
                    expert_idx = top_k_indices[i, j, k].item()
                    prob = top_k_probs[i, j, k].item()
                    dispatch_tensor[i, j, expert_idx, 0] = 1.0
                    combine_tensor[i, j, expert_idx, 0] = prob
        
        # 计算辅助损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss


class TimeMoERouter(BaseRouter):
    """
    Time-MoE路由器：专门为时间序列预测设计的MoE框架
    参考：https://github.com/Time-MoE/Time-MoE
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2,
        sequence_length: int = 512,
        use_temporal_attention: bool = True
    ):
        super().__init__(hidden_size, num_experts, top_k)
        
        self.sequence_length = sequence_length
        self.use_temporal_attention = use_temporal_attention
        
        # 时间编码器
        self.temporal_encoder = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),  # +2 for timestamp and seasonality
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 时间感知路由器
        self.temporal_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
        # 时间注意力机制
        if use_temporal_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        
        # 位置编码
        self.position_embedding = nn.Embedding(sequence_length, hidden_size)
        
        # 季节性编码器
        self.seasonal_encoder = nn.Sequential(
            nn.Linear(1, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 4)
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        time_info: Optional[Dict[str, torch.Tensor]] = None,
        expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Time-MoE路由逻辑
        
        Args:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            time_info: 时间信息字典，包含时间戳和季节性信息
            expert_mask: 专家可用性掩码
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 生成时间特征
        if time_info is None:
            timestamps = torch.arange(seq_len, device=hidden_states.device).float()
            seasonality = torch.sin(timestamps * 2 * torch.pi / 24)  # 24小时周期
        else:
            timestamps = time_info.get('timestamps', torch.arange(seq_len, device=hidden_states.device).float())
            seasonality = time_info.get('seasonality', torch.sin(timestamps * 2 * torch.pi / 24))
        
        # 扩展时间特征到batch维度
        timestamps = timestamps.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
        seasonality = seasonality.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
        
        # 位置编码
        positions = torch.arange(seq_len, device=hidden_states.device)
        position_embeddings = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 季节性编码
        seasonal_embeddings = self.seasonal_encoder(seasonality.unsqueeze(-1))  # [batch_size, seq_len, hidden_size//4]
        seasonal_embeddings = seasonal_embeddings.expand(-1, -1, hidden_size)  # 扩展到完整维度
        
        # 组合时间特征
        time_features = torch.stack([timestamps, seasonality], dim=-1)  # [batch_size, seq_len, 2]
        combined_features = torch.cat([hidden_states, time_features], dim=-1)  # [batch_size, seq_len, hidden_size+2]
        
        # 时间编码
        temporal_encoded = self.temporal_encoder(combined_features)
        
        # 添加位置编码
        temporal_encoded = temporal_encoded + position_embeddings + seasonal_embeddings
        
        # 时间注意力
        if self.use_temporal_attention:
            temporal_encoded, _ = self.temporal_attention(
                temporal_encoded, temporal_encoded, temporal_encoded
            )
        
        # 时间感知路由
        router_logits = self.temporal_router(temporal_encoded)
        
        # 应用专家掩码
        if expert_mask is not None:
            router_logits = router_logits + (1 - expert_mask.unsqueeze(0).unsqueeze(0)) * -1e9
        
        # 路由概率
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 获取top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 创建调度和组合张量
        capacity = self._compute_capacity(batch_size, seq_len)
        
        dispatch_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        combine_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        
        # 分配逻辑
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(self.top_k):
                    expert_idx = top_k_indices[i, j, k].item()
                    prob = top_k_probs[i, j, k].item()
                    dispatch_tensor[i, j, expert_idx, 0] = 1.0
                    combine_tensor[i, j, expert_idx, 0] = prob
        
        # 计算辅助损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss


class FastMoERouter(BaseRouter):
    """
    FastMoE路由器：高性能MoE实现，支持分布式训练和推理
    参考：https://github.com/laekov/fastmoe
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2,
        world_size: int = 1,
        gate_type: str = 'top'
    ):
        super().__init__(hidden_size, num_experts, top_k)
        
        self.world_size = world_size
        self.gate_type = gate_type
        
        # FastMoE风格的路由器
        self.fast_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
        # 负载均衡损失
        self.load_balancing_loss_weight = 0.01
        
        # 分布式相关
        self.experts_per_rank = num_experts // world_size
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        FastMoE路由逻辑
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # FastMoE风格的路由计算
        router_logits = self.fast_router(hidden_states)
        
        # 应用专家掩码
        if expert_mask is not None:
            router_logits = router_logits + (1 - expert_mask.unsqueeze(0).unsqueeze(0)) * -1e9
        
        # 路由概率
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 获取top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 创建调度和组合张量
        capacity = self._compute_capacity(batch_size, seq_len)
        
        dispatch_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        combine_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        
        # 分配逻辑
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(self.top_k):
                    expert_idx = top_k_indices[i, j, k].item()
                    prob = top_k_probs[i, j, k].item()
                    dispatch_tensor[i, j, expert_idx, 0] = 1.0
                    combine_tensor[i, j, expert_idx, 0] = prob
        
        # FastMoE风格的负载均衡损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = self.load_balancing_loss_weight * torch.sum(
            router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9)
        )
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss


class MixtureOfExpertsRouter(BaseRouter):
    """
    Mixture of Experts路由器：通用的MoE架构
    参考：https://github.com/lucidrains/mixture-of-experts
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2,
        capacity_factor_train: float = 1.25,
        capacity_factor_eval: float = 2.0,
        loss_coef: float = 1e-2
    ):
        super().__init__(hidden_size, num_experts, top_k)
        
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.loss_coef = loss_coef
        
        # MoE路由器
        self.moe_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
        # 辅助损失权重
        self.aux_loss_weight = 1e-2
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Mixture of Experts路由逻辑
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 根据训练状态选择容量因子
        capacity_factor = self.capacity_factor_train if training else self.capacity_factor_eval
        
        # MoE路由计算
        router_logits = self.moe_router(hidden_states)
        
        # 应用专家掩码
        if expert_mask is not None:
            router_logits = router_logits + (1 - expert_mask.unsqueeze(0).unsqueeze(0)) * -1e9
        
        # 路由概率
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 获取top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 计算容量
        capacity = int(batch_size * seq_len * capacity_factor * self.top_k / self.num_experts)
        
        # 创建调度和组合张量
        dispatch_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        combine_tensor = torch.zeros(
            batch_size, seq_len, self.num_experts, capacity,
            device=hidden_states.device
        )
        
        # 分配逻辑
        for i in range(batch_size):
            for j in range(seq_len):
                for k in range(self.top_k):
                    expert_idx = top_k_indices[i, j, k].item()
                    prob = top_k_probs[i, j, k].item()
                    dispatch_tensor[i, j, expert_idx, 0] = 1.0
                    combine_tensor[i, j, expert_idx, 0] = prob
        
        # MoE风格的辅助损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = self.aux_loss_weight * torch.sum(
            router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9)
        )
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss


# MoE策略工厂函数
def create_moe_router(
    router_type: str,
    hidden_size: int,
    num_experts: int,
    **kwargs
) -> BaseRouter:
    """
    创建指定类型的MoE路由器
    
    Args:
        router_type: 路由器类型 ('flex', 'time', 'fast', 'mixture', 'base')
        hidden_size: 隐藏层大小
        num_experts: 专家数量
        **kwargs: 其他参数
    
    Returns:
        BaseRouter: 对应的路由器实例
    """
    router_map = {
        'flex': FlexMoERouter,
        'time': TimeMoERouter,
        'fast': FastMoERouter,
        'mixture': MixtureOfExpertsRouter,
        'base': BaseRouter
    }
    
    if router_type not in router_map:
        raise ValueError(f"Unsupported router type: {router_type}. "
                        f"Supported types: {list(router_map.keys())}")
    
    return router_map[router_type](hidden_size, num_experts, **kwargs) 