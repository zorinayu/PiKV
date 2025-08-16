"""
Unified MoE (Mixture of Experts) Implementation
Consolidates all routing strategies and expert management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union
from config import config

class BaseRouter(nn.Module):
    """Base router class for MoE routing logic"""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Route input to experts"""
        batch_size, seq_len, _ = x.shape
        
        # Calculate routing logits
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Calculate auxiliary loss for load balancing
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        return top_k_indices, top_k_probs, float(aux_loss.item())


class FlexMoERouter(BaseRouter):
    """Flex-MoE router for multimodal learning"""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 4):
        super().__init__(hidden_size, num_experts, top_k)
        
        # Modality-specific routers
        self.modality_routers = nn.ModuleDict({
            'image': nn.Linear(hidden_size, num_experts),
            'text': nn.Linear(hidden_size, num_experts),
            'audio': nn.Linear(hidden_size, num_experts)
        })
        
        # Generalized router
        self.generalized_router = nn.Linear(hidden_size, num_experts)
    
    def forward(self, x: torch.Tensor, modality_info: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if modality_info is None:
            return super().forward(x)
        
        # Combine modality-specific and generalized routing
        combined_logits = self.generalized_router(x)
        for modality_name, modality_data in modality_info.items():
            if modality_name in self.modality_routers:
                modality_logits = self.modality_routers[modality_name](modality_data)
                combined_logits = combined_logits + modality_logits
        
        # Apply routing logic
        router_probs = F.softmax(combined_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Calculate auxiliary loss
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        return top_k_indices, top_k_probs, float(aux_loss.item())


class TimeMoERouter(BaseRouter):
    """Time-MoE router for time series prediction"""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__(hidden_size, num_experts, top_k)
        
        # Temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Position embedding
        self.position_embedding = nn.Embedding(2048, hidden_size)
    
    def forward(self, x: torch.Tensor, time_info: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
        batch_size, seq_len, _ = x.shape
        
        # Generate temporal features
        if time_info is None:
            timestamps = torch.arange(seq_len, device=x.device).float()
            seasonality = torch.sin(timestamps * 2 * torch.pi / 24)
        else:
            timestamps = time_info.get('timestamps', torch.arange(seq_len, device=x.device).float())
            seasonality = time_info.get('seasonality', torch.sin(timestamps * 2 * torch.pi / 24))
        
        # Combine temporal features
        time_features = torch.stack([timestamps, seasonality], dim=-1)
        time_features = time_features.unsqueeze(0).expand(batch_size, -1, -1)
        combined_features = torch.cat([x, time_features], dim=-1)
        
        # Encode temporal information
        temporal_encoded = self.temporal_encoder(combined_features)
        
        # Add position embeddings
        positions = torch.arange(seq_len, device=x.device)
        position_embeddings = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        temporal_encoded = temporal_encoded + position_embeddings
        
        # Apply routing
        return super().forward(temporal_encoded)


class MoE(nn.Module):
    """Unified Mixture of Experts implementation"""
    
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2,
        router_type: str = 'base',
        expert_size: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_size = expert_size or hidden_size * 2
        
        # Create router based on type
        if router_type == 'flex':
            self.router = FlexMoERouter(hidden_size, num_experts, top_k)
        elif router_type == 'time':
            self.router = TimeMoERouter(hidden_size, num_experts, top_k)
        else:
            self.router = BaseRouter(hidden_size, num_experts, top_k)
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, self.expert_size),
                nn.ReLU(),
                nn.Linear(self.expert_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, float]:
        """Forward pass through MoE"""
        batch_size, seq_len, _ = x.shape
        
        # Route input to experts
        expert_indices, expert_weights, aux_loss = self.router(x, **kwargs)
        
        # Simplified expert processing - use first expert for all inputs
        expert_output = self.experts[0](x)
        
        # Apply output projection
        output = self.output_projection(expert_output)
        
        return output, float(aux_loss)


class PiKVMoE(MoE):
    """PiKV-specific MoE implementation with additional features"""
    
    def __init__(
        self, 
        rank: int = 4, 
        alpha: float = 1.0,
        use_distillation: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rank = rank
        self.alpha = alpha
        self.use_distillation = use_distillation
        
        # LoRA layers for experts
        self.expert_lora = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, rank),
                nn.Linear(rank, self.hidden_size)
            ) for _ in range(self.num_experts)
        ])
        
        # Distillation components
        if use_distillation:
            self.teacher_projection = nn.Linear(self.hidden_size, self.hidden_size * 2)
            self.student_projection = nn.Linear(self.hidden_size, self.hidden_size * 2)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, float]:
        """Forward pass with LoRA and distillation"""
        # Standard MoE forward pass
        output, aux_loss = super().forward(x, **kwargs)
        
        # Apply LoRA to expert outputs
        for i, lora in enumerate(self.expert_lora):
            if i < len(self.experts):
                output = output + self.alpha * lora(output)
        
        # Apply distillation if enabled
        if self.use_distillation and self.training:
            teacher_output = self.teacher_projection(x)
            student_output = self.student_projection(output)
            distillation_loss = F.mse_loss(student_output, teacher_output)
            aux_loss = aux_loss + distillation_loss
        
        return output, aux_loss


# Factory function for creating MoE instances
def create_moe(
    moe_type: str = 'base',
    hidden_size: int = 1024,
    num_experts: int = 8,
    **kwargs
) -> MoE:
    """Create MoE instance based on type"""
    
    moe_map = {
        'base': MoE,
        'pikv': PiKVMoE,
        'flex': lambda **kw: MoE(router_type='flex', **kw),
        'time': lambda **kw: MoE(router_type='time', **kw)
    }
    
    if moe_type not in moe_map:
        raise ValueError(f"Unsupported MoE type: {moe_type}")
    
    return moe_map[moe_type](
        hidden_size=hidden_size,
        num_experts=num_experts,
        **kwargs
    )
