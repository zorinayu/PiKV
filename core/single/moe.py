"""
Enhanced MoE (Mixture of Experts) Implementation
Advanced routing strategies with normalization, LoRA, EPLB, and hierarchical routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union
from config import config
import math

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
        
        # Layer normalization for router input
        self.router_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Route input to experts"""
        batch_size, seq_len, _ = x.shape
        
        # Apply normalization
        x_norm = self.router_norm(x)
        
        # Calculate routing logits
        router_logits = self.router(x_norm)  # [batch_size, seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Calculate auxiliary loss for load balancing
        router_prob_per_expert = router_probs.mean(dim=[0, 1])
        aux_loss = torch.sum(router_prob_per_expert * torch.log(router_prob_per_expert * self.num_experts + 1e-9))
        
        return top_k_indices, top_k_probs, float(aux_loss.item())


class EPLBRouter(BaseRouter):
    """EPLB (Expert Parallel Load Balancing) Router"""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, balance_coefficient: float = 0.01):
        super().__init__(hidden_size, num_experts, top_k)
        self.balance_coefficient = balance_coefficient
        
        # Load balancing network
        self.load_balancer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
        # Expert load tracking
        self.register_buffer('expert_loads', torch.zeros(num_experts))
        self.register_buffer('total_load', torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """EPLB routing with load balancing"""
        batch_size, seq_len, _ = x.shape
        
        # Apply normalization
        x_norm = self.router_norm(x)
        
        # Main routing
        router_logits = self.router(x_norm)
        
        # Load balancing routing
        load_balance_logits = self.load_balancer(x_norm)
        
        # Combine routing with load balancing
        combined_logits = router_logits + self.balance_coefficient * load_balance_logits
        
        # Apply softmax
        router_probs = F.softmax(combined_logits, dim=-1)
        
        # Get top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Update expert loads
        self._update_loads(router_probs)
        
        # Calculate auxiliary loss
        aux_loss = self._get_load_balance_loss()
        
        return top_k_indices, top_k_probs, float(aux_loss.item())
    
    def _update_loads(self, routing_weights):
        """Update expert load statistics"""
        expert_loads = routing_weights.sum(dim=[0, 1])
        self.expert_loads += expert_loads
        self.total_load += expert_loads.sum()
    
    def _get_load_balance_loss(self):
        """Calculate load balancing loss"""
        avg_load = self.total_load / self.num_experts
        load_imbalance = torch.abs(self.expert_loads - avg_load)
        return self.balance_coefficient * load_imbalance.mean()


class HierarchicalRouter(BaseRouter):
    """Hierarchical Router for large-scale expert systems"""
    
    def __init__(self, hidden_size: int, num_experts: int, num_groups: int = 4, group_top_k: int = 1):
        super().__init__(hidden_size, num_groups, group_top_k)
        self.num_groups = num_groups
        self.group_top_k = group_top_k
        self.experts_per_group = num_experts // num_groups
        
        # Group-level router
        self.group_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_groups)
        )
        
        # Expert-level routers (one per group)
        self.expert_routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, self.experts_per_group)
            ) for _ in range(num_groups)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Hierarchical routing"""
        batch_size, seq_len, _ = x.shape
        
        # Apply normalization
        x_norm = self.router_norm(x)
        
        # First stage: select expert groups
        group_logits = self.group_router(x_norm)
        group_probs = F.softmax(group_logits, dim=-1)
        
        # Select top-k groups
        top_k_group_probs, top_k_group_indices = torch.topk(
            group_probs, k=min(self.group_top_k, self.num_groups), dim=-1
        )
        
        # Second stage: select experts within groups
        all_expert_probs = []
        all_expert_indices = []
        
        for group_idx in range(self.num_groups):
            expert_logits = self.expert_routers[group_idx](x_norm)
            expert_probs = F.softmax(expert_logits, dim=-1)
            
            # Convert to global expert indices
            global_expert_indices = torch.arange(
                group_idx * self.experts_per_group,
                (group_idx + 1) * self.experts_per_group,
                device=x.device
            )
            
            all_expert_probs.append(expert_probs)
            all_expert_indices.append(global_expert_indices)
        
        # Combine group and expert probabilities
        router_probs = torch.zeros(batch_size, seq_len, self.num_experts, device=x.device)
        
        for group_idx in range(self.num_groups):
            group_weight = group_probs[:, :, group_idx:group_idx+1]
            expert_probs = all_expert_probs[group_idx]
            
            start_idx = group_idx * self.experts_per_group
            end_idx = (group_idx + 1) * self.experts_per_group
            
            router_probs[:, :, start_idx:end_idx] = group_weight * expert_probs
        
        # Get final top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Calculate auxiliary loss
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
        
        # Apply normalization
        x_norm = self.router_norm(x)
        
        # Combine modality-specific and generalized routing
        combined_logits = self.generalized_router(x_norm)
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
        
        # Apply routing with normalization
        return super().forward(temporal_encoded)


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) Layer"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Scaling factor
        self.scaling = alpha / rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


class MoE(nn.Module):
    """Enhanced Mixture of Experts implementation"""
    
    def __init__(
        self, 
        hidden_size: int, 
        num_experts: int, 
        top_k: int = 2,
        router_type: str = 'base',
        expert_size: Optional[int] = None,
        use_normalization: bool = True,
        use_lora: bool = False,
        lora_rank: int = 16
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_size = expert_size or hidden_size * 2
        self.use_normalization = use_normalization
        self.use_lora = use_lora
        
        # Create router based on type
        if router_type == 'eplb':
            self.router = EPLBRouter(hidden_size, num_experts, top_k)
        elif router_type == 'hierarchical':
            self.router = HierarchicalRouter(hidden_size, num_experts)
        elif router_type == 'flex':
            self.router = FlexMoERouter(hidden_size, num_experts, top_k)
        elif router_type == 'time':
            self.router = TimeMoERouter(hidden_size, num_experts, top_k)
        else:
            self.router = BaseRouter(hidden_size, num_experts, top_k)
        
        # Input normalization
        if use_normalization:
            self.input_norm = nn.LayerNorm(hidden_size)
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, self.expert_size),
                nn.ReLU(),
                nn.Linear(self.expert_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        
        # LoRA layers for experts
        if use_lora:
            self.expert_lora = nn.ModuleList([
                LoRALayer(hidden_size, hidden_size, lora_rank)
                for _ in range(num_experts)
            ])
        
        # Output projection and normalization
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        if use_normalization:
            self.output_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, float]:
        """Forward pass through MoE"""
        batch_size, seq_len, _ = x.shape
        
        # Input normalization
        if self.use_normalization:
            x = self.input_norm(x)
        
        # Route input to experts
        expert_indices, expert_weights, aux_loss = self.router(x, **kwargs)
        
        # Simplified expert processing - use first expert for all inputs
        expert_output = self.experts[0](x)
        
        # Apply LoRA if enabled
        if self.use_lora:
            expert_output = expert_output + self.expert_lora[0](x)
        
        # Apply output projection and normalization
        output = self.output_projection(expert_output)
        if self.use_normalization:
            output = self.output_norm(output)
        
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
        
        # Enhanced LoRA layers for experts
        if self.use_lora:
            self.expert_lora = nn.ModuleList([
                LoRALayer(self.hidden_size, self.hidden_size, rank, alpha)
                for _ in range(self.num_experts)
            ])
        
        # Distillation components
        if use_distillation:
            self.teacher_projection = nn.Linear(self.hidden_size, self.hidden_size * 2)
            self.student_projection = nn.Linear(self.hidden_size, self.hidden_size * 2)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, float]:
        """Forward pass with LoRA and distillation"""
        # Standard MoE forward pass
        output, aux_loss = super().forward(x, **kwargs)
        
        # Apply enhanced LoRA to expert outputs
        if self.use_lora:
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
        'eplb': lambda **kw: MoE(router_type='eplb', **kw),
        'hierarchical': lambda **kw: MoE(router_type='hierarchical', **kw),
        'flex': lambda **kw: MoE(router_type='flex', **kw),
        'time': lambda **kw: MoE(router_type='time', **kw)
    }
    
    if moe_type not in moe_map:
        raise ValueError(f"Unsupported MoE type: {moe_type}. "
                        f"Supported types: {list(moe_map.keys())}")
    
    return moe_map[moe_type](
        hidden_size=hidden_size,
        num_experts=num_experts,
        **kwargs
    )
