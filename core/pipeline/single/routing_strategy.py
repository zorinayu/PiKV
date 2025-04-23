import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AdaptiveRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, temperature: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        
        # Initialize router parameters
        self.router = nn.Linear(hidden_size, num_experts)
        self.importance = nn.Linear(hidden_size, 1)
        
        # Initialize loss coefficients
        self.register_buffer('load_balancing_loss_coef', torch.tensor(0.01))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate routing logits
        routing_logits = self.router(x)
        
        # Calculate importance scores
        importance = torch.sigmoid(self.importance(x))
        
        # Apply temperature scaling
        routing_logits = routing_logits / self.temperature
        
        # Calculate routing probabilities
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Get top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        
        # Calculate load balancing loss
        expert_mask = torch.zeros(self.num_experts, device=x.device)
        expert_mask.scatter_add_(0, top_k_indices.view(-1), top_k_weights.view(-1))
        expert_mask = expert_mask / (top_k_weights.sum() + 1e-6)
        load_balancing_loss = torch.sum(expert_mask * torch.log(expert_mask + 1e-6))
        
        return routing_probs, top_k_indices, top_k_weights, load_balancing_loss, importance 