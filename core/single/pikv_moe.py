import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])

    def forward(self, x):
        return F.relu(self.dense(x))

class PiKVMoE(nn.Module):
    def __init__(self):
        super(PiKVMoE, self).__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(config['num_experts'])])
        self.gate = nn.Linear(config['hidden_size'], config['num_experts'])

        # Cache size allocation for each layer
        self.cache_sizes = self.pyramidal_cache_allocation()

    def pyramidal_cache_allocation(self):
        """
        Calculate the cache size for each layer using the pyramidal allocation policy.
        """
        C1 = config['kv_cache_size']
        d = config['cache_decrement']
        return [C1 - (i - 1) * d for i in range(1, config['num_layers'] + 1)]

    def forward(self, x):
        # Calculate gate scores
        gate_scores = self.gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)

        # Select top expert based on gate probabilities
        expert_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_output += expert(x) * gate_probs[:, i].unsqueeze(-1)

        return expert_output
