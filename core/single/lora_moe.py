import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config
from .lora import LoRALayer, LoRAExpert

class LoRAMoE(nn.Module):
    def __init__(self, rank=4, alpha=1.0):
        super(LoRAMoE, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_experts = config['num_experts']
        
        # Initialize experts with LoRA adaptation
        self.experts = nn.ModuleList([
            LoRAExpert(self.hidden_size, rank=rank, alpha=alpha)
            for _ in range(self.num_experts)
        ])
        
        # Initialize router
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # Ensure input has correct shape [batch_size, seq_len, hidden_size]
        if len(x.shape) == 2:  # [batch_size, hidden_size]
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
        elif len(x.shape) == 1:  # [hidden_size]
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        
        # Calculate routing weights using mean across sequence length
        x_mean = x.mean(dim=1)  # [batch_size, hidden_size]
        routing_weights = self.router(x_mean)  # [batch_size, num_experts]
        
        # Initialize output tensor
        expert_output = torch.zeros_like(x)
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Get expert output with LoRA adaptation
            expert_output_i = expert(x)  # [batch_size, seq_len, hidden_size]
            
            # Reshape routing weights to match expert output dimensions
            routing_weights_i = routing_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
            
            # Add to final output weighted by routing probabilities
            expert_output += expert_output_i * routing_weights_i
        
        return expert_output 