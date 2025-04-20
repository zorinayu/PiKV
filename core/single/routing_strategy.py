import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExpertRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2, temperature=1.0):
        super(ExpertRouter, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        
        # Initialize routing network
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Initialize expert capacity
        self.expert_capacity = nn.Parameter(torch.ones(num_experts))
        
    def forward(self, x, expert_mask=None):
        """
        Route input to experts.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            expert_mask: Optional mask for expert availability of shape [num_experts]
        
        Returns:
            routing_weights: Routing weights of shape [batch_size, seq_len, num_experts]
            expert_indices: Selected expert indices of shape [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Get routing logits
        routing_logits = self.router(x)
        
        # Apply temperature scaling
        routing_logits = routing_logits / self.temperature
        
        # Apply expert mask if provided
        if expert_mask is not None:
            routing_logits = routing_logits + expert_mask.unsqueeze(0).unsqueeze(0)
        
        # Get top-k experts
        routing_weights = F.softmax(routing_logits, dim=-1)
        top_k_weights, expert_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, expert_indices, top_k_weights

class LoadBalancingRouter(ExpertRouter):
    def __init__(self, hidden_size, num_experts, top_k=2, temperature=1.0, balance_weight=0.01):
        super(LoadBalancingRouter, self).__init__(hidden_size, num_experts, top_k, temperature)
        self.balance_weight = balance_weight
        
    def forward(self, x, expert_mask=None):
        routing_weights, expert_indices, top_k_weights = super().forward(x, expert_mask)
        
        # Calculate load balancing loss
        expert_usage = routing_weights.mean(dim=0).mean(dim=0)  # [num_experts]
        load_balance_loss = self.balance_weight * (
            expert_usage.std() / expert_usage.mean()
        )
        
        return routing_weights, expert_indices, top_k_weights, load_balance_loss

class ImportanceAwareRouter(ExpertRouter):
    def __init__(self, hidden_size, num_experts, top_k=2, temperature=1.0):
        super(ImportanceAwareRouter, self).__init__(hidden_size, num_experts, top_k, temperature)
        
        # Initialize importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, expert_mask=None):
        # Predict importance scores
        importance = self.importance_predictor(x)  # [batch_size, seq_len, 1]
        
        # Get base routing weights
        routing_weights, expert_indices, top_k_weights = super().forward(x, expert_mask)
        
        # Adjust routing weights by importance
        routing_weights = routing_weights * importance
        
        # Renormalize
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, expert_indices, top_k_weights, importance

class AdaptiveRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2, temperature=1.0):
        super(AdaptiveRouter, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        
        # Initialize multiple routing strategies
        self.load_balancing_router = LoadBalancingRouter(
            hidden_size, num_experts, top_k, temperature
        )
        self.importance_router = ImportanceAwareRouter(
            hidden_size, num_experts, top_k, temperature
        )
        
        # Initialize router selector
        self.router_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, expert_mask=None):
        # Get router selection weights
        router_weights = self.router_selector(x.mean(dim=1))  # [batch_size, 2]
        
        # Get routing from both strategies
        lb_weights, lb_indices, lb_top_k, lb_loss = self.load_balancing_router(x, expert_mask)
        imp_weights, imp_indices, imp_top_k, importance = self.importance_router(x, expert_mask)
        
        # Combine routing weights
        routing_weights = (
            router_weights[:, 0:1, None] * lb_weights +
            router_weights[:, 1:2, None] * imp_weights
        )
        
        # Get top-k experts
        top_k_weights, expert_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, expert_indices, top_k_weights, lb_loss, importance 