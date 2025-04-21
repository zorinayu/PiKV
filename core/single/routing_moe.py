import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config

class AdaptiveRouter(nn.Module):
    def __init__(self, hidden_size, num_experts):
        super(AdaptiveRouter, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        
        # Main routing network
        self.main_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Auxiliary routing network for load balancing
        self.aux_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Expert load tracking
        self.register_buffer('expert_loads', torch.zeros(num_experts))
        self.register_buffer('total_load', torch.tensor(0.0))
        
        # Load balancing parameters
        self.load_balance_weight = 0.1
        self.load_balance_threshold = 0.1
    
    def update_loads(self, routing_weights):
        # Update expert loads based on routing weights
        self.expert_loads += routing_weights.sum(dim=0)
        self.total_load += routing_weights.sum()
    
    def get_load_balance_loss(self):
        # Calculate load balancing loss
        avg_load = self.total_load / self.num_experts
        load_imbalance = torch.abs(self.expert_loads - avg_load)
        return self.load_balance_weight * load_imbalance.mean()
    
    def forward(self, x):
        # Get main routing weights
        main_weights = self.main_router(x)
        
        # Get auxiliary routing weights
        aux_weights = self.aux_router(x)
        
        # Combine weights based on load balancing
        combined_weights = main_weights * (1 - self.load_balance_weight) + \
                         aux_weights * self.load_balance_weight
        
        # Normalize combined weights
        routing_weights = F.softmax(combined_weights, dim=-1)
        
        # Update expert loads
        self.update_loads(routing_weights)
        
        return routing_weights

class RoutingMoE(nn.Module):
    def __init__(self):
        super(RoutingMoE, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_experts = config['num_experts']
        
        # Initialize experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU()
            ) for _ in range(self.num_experts)
        ])
        
        # Initialize adaptive router
        self.router = AdaptiveRouter(self.hidden_size, self.num_experts)
    
    def forward(self, x):
        # Calculate routing weights
        routing_weights = self.router(x)  # [batch_size, num_experts]
        
        # Initialize output tensor
        expert_output = torch.zeros_like(x)
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Get expert output
            expert_output_i = expert(x)
            
            # Add to final output weighted by routing probabilities
            expert_output += expert_output_i * routing_weights[:, i].unsqueeze(-1)
        
        return expert_output
    
    def get_load_balance_loss(self):
        # Get load balancing loss from router
        return self.router.get_load_balance_loss() 