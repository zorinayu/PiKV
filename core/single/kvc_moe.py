import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config
from .kv_cache_compression import KVCacheCompressor

class KVCache(nn.Module):
    def __init__(self, size, hidden_size):
        super(KVCache, self).__init__()
        self.size = size
        self.hidden_size = hidden_size
        
        # Initialize tensors
        self.register_buffer('keys', torch.zeros(size, hidden_size))
        self.register_buffer('values', torch.zeros(size, hidden_size))
        self.register_buffer('importance', torch.zeros(size))
        
        # Initialize compressor
        self.compressor = KVCacheCompressor(
            hidden_size=hidden_size,
            compression_type='pyramid',
            compression_ratio=0.5
        )
    
    def update(self, idx, key, value, importance):
        # Update cache at the specified index
        self.keys[idx] = key
        self.values[idx] = value
        self.importance[idx] = importance
    
    def get_all(self):
        # Apply compression to cached values
        compressed_keys, compressed_values = self.compressor(
            self.keys.unsqueeze(0),
            self.values.unsqueeze(0),
            self.importance.unsqueeze(0)
        )
        return compressed_values.squeeze(0)

class KVCacheMoE(nn.Module):
    def __init__(self):
        super(KVCacheMoE, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_experts = config['num_experts']
        
        # Initialize experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU()
            ) for _ in range(self.num_experts)
        ])
        
        # Initialize router
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Initialize KV caches for each expert
        self.kv_caches = nn.ModuleList([
            KVCache(config['kv_cache_size'], self.hidden_size)
            for _ in range(self.num_experts)
        ])
        
        # Cache pointers for each expert
        self.register_buffer('cache_ptrs', torch.zeros(self.num_experts, dtype=torch.long))
    
    def update_cache(self, expert_idx, key, value, importance):
        cache = self.kv_caches[expert_idx]
        ptr = self.cache_ptrs[expert_idx]
        
        # Update cache with new key-value pair
        cache.update(ptr, key, value, importance)
        
        # Update pointer
        self.cache_ptrs[expert_idx] = (ptr + 1) % cache.size
    
    def forward(self, x):
        # Calculate routing weights
        routing_weights = self.router(x)  # [batch_size, num_experts]
        
        # Initialize output tensor
        expert_output = torch.zeros_like(x)
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Get expert output
            expert_output_i = expert(x)
            
            # Get cached values
            cached_values = self.kv_caches[i].get_all()
            
            # Combine with cached values
            if cached_values is not None:
                expert_output_i = expert_output_i + cached_values.unsqueeze(0)
            
            # Update cache with new values
            self.update_cache(i, x.detach(), expert_output_i.detach(), routing_weights[:, i].detach())
            
            # Add to final output weighted by routing probabilities
            expert_output += expert_output_i * routing_weights[:, i].unsqueeze(-1)
        
        return expert_output 