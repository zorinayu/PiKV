import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config

class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])

    def forward(self, x):
        return F.relu(self.dense(x))

class StandardMoE(nn.Module):
    def __init__(self):
        super(StandardMoE, self).__init__()
        
        # Add embedding layer to handle token IDs
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        
        self.experts = nn.ModuleList([Expert() for _ in range(config['num_experts'])])
        
        # Projection to vocabulary size
        self.vocab_proj = nn.Linear(config['hidden_size'], config['vocab_size'])

    def forward(self, x):
        # Handle both token IDs (2D) and embeddings (3D) as input
        if len(x.shape) == 2:  # Token IDs: [batch_size, seq_len]
            x = self.embedding(x)  # Convert to embeddings: [batch_size, seq_len, hidden_size]
        elif len(x.shape) == 3:  # Already embeddings: [batch_size, seq_len, hidden_size]
            pass
        else:
            raise ValueError(f"Input must be 2D (token IDs) or 3D (embeddings), got shape: {x.shape}")
        
        # Directly pass input to all experts
        expert_output = torch.zeros_like(x)
        for expert in self.experts:
            expert_output += expert(x)
        
        # Project to vocabulary size
        logits = self.vocab_proj(expert_output)
        return logits
