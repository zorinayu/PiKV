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
        self.capacity_factor = capacity_factor
        self.expert_backend = expert_backend
        
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
            router_logits = router_logits + (1 - expert_mask.unsqueeze(0).unsqueeze(0)) * -1e9
            
        # 路由概率
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
        
        # 获取top_k专家
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)  # [batch_size, seq_len, top_k]
        
        # 重新归一化top_k概率
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # 计算专家容量
        capacity = self._compute_capacity(batch_size, seq_len)
        
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
                    expert_idx = int(top_k_indices[i, j, k].item())
                    prob = float(top_k_probs[i, j, k].item())
                    dispatch_tensor[i, j, expert_idx, 0] = 1.0
                    combine_tensor[i, j, expert_idx, 0] = prob
        
        # 计算辅助损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = float(torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9)).item())
        
        return dispatch_tensor, combine_tensor, router_probs, aux_loss


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
                    expert_idx = int(top_k_indices[i, j, k].item())
                    prob = float(top_k_probs[i, j, k].item())
                    dispatch_tensor[i, j, expert_idx, 0] = 1.0
                    combine_tensor[i, j, expert_idx, 0] = prob
        
        # 计算辅助损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = float(torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9)).item())
        
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
        # 使用线性层将季节性编码扩展到完整维度
        seasonal_expand = nn.Linear(hidden_size // 4, hidden_size, device=hidden_states.device)
        seasonal_embeddings = seasonal_expand(seasonal_embeddings)  # [batch_size, seq_len, hidden_size]
        
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
                    expert_idx = int(top_k_indices[i, j, k].item())
                    prob = float(top_k_probs[i, j, k].item())
                    dispatch_tensor[i, j, expert_idx, 0] = 1.0
                    combine_tensor[i, j, expert_idx, 0] = prob
        
        # 计算辅助损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = float(torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9)).item())
        
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
                    expert_idx = int(top_k_indices[i, j, k].item())
                    prob = float(top_k_probs[i, j, k].item())
                    dispatch_tensor[i, j, expert_idx, 0] = 1.0
                    combine_tensor[i, j, expert_idx, 0] = prob
        
        # FastMoE风格的负载均衡损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = self.load_balancing_loss_weight * float(torch.sum(
            router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9)
        ).item())
        
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
                    expert_idx = int(top_k_indices[i, j, k].item())
                    prob = float(top_k_probs[i, j, k].item())
                    dispatch_tensor[i, j, expert_idx, 0] = 1.0
                    combine_tensor[i, j, expert_idx, 0] = prob
        
        # MoE风格的辅助损失
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = self.aux_loss_weight * float(torch.sum(
            router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9)
        ).item())
        
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


# 使用示例
if __name__ == "__main__":
    # 测试各种MoE路由器
    hidden_size = 1024
    num_experts = 8
    batch_size = 4
    seq_len = 128
    
    # 创建测试数据
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 测试Flex-MoE路由器
    flex_router = create_moe_router('flex', hidden_size, num_experts, top_k=4)
    modality_info = {
        'image': torch.randn(batch_size, seq_len, hidden_size),
        'genomic': torch.randn(batch_size, seq_len, hidden_size)
    }
    dispatch, combine, probs, loss = flex_router(hidden_states, modality_info)
    print(f"Flex-MoE: dispatch shape {dispatch.shape}, loss {loss}")
    
    # 测试Time-MoE路由器
    time_router = create_moe_router('time', hidden_size, num_experts, top_k=2)
    time_info = {
        'timestamps': torch.arange(seq_len).float(),
        'seasonality': torch.sin(torch.arange(seq_len) * 2 * torch.pi / 24)
    }
    dispatch, combine, probs, loss = time_router(hidden_states, time_info)
    print(f"Time-MoE: dispatch shape {dispatch.shape}, loss {loss}")
    
    # 测试FastMoE路由器
    fast_router = create_moe_router('fast', hidden_size, num_experts, top_k=2)
    dispatch, combine, probs, loss = fast_router(hidden_states)
    print(f"FastMoE: dispatch shape {dispatch.shape}, loss {loss}")
    
    # 测试Mixture of Experts路由器
    mixture_router = create_moe_router('mixture', hidden_size, num_experts, top_k=2)
    dispatch, combine, probs, loss = mixture_router(hidden_states, training=True)
    print(f"Mixture of Experts: dispatch shape {dispatch.shape}, loss {loss}")
