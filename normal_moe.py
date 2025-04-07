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

class StandardMoE(nn.Module):
    def __init__(self):
        super(StandardMoE, self).__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(config['num_experts'])])

    def forward(self, x):
        # Directly pass input to all experts
        expert_output = torch.zeros_like(x)
        for expert in self.experts:
            expert_output += expert(x)
        return expert_output
