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
        
        # Initialize tensors with correct dimensions
        self.register_buffer('keys', torch.zeros(size, hidden_size, device=config['device']))
        self.register_buffer('values', torch.zeros(size, hidden_size, device=config['device']))
        self.register_buffer('importance', torch.zeros(size, device=config['device']))
        
        # Initialize compressor
        self.compressor = KVCacheCompressor(
            hidden_size=hidden_size,
            compression_type='pyramid',
            compression_ratio=0.5
        )
    
    def update(self, idx, key, value, importance):
        # Ensure input tensors have correct shape and device
        key = key.to(self.keys.device)
        value = value.to(self.values.device)
        importance = importance.to(self.importance.device)
        
        # Reshape tensors to match cache dimensions
        if len(key.shape) == 3:  # [batch_size, seq_len, hidden_size]
            # Take mean across batch and sequence dimensions
            key = key.mean(dim=0).mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0).mean(dim=0)  # [hidden_size]
            importance = importance.mean()  # scalar
        elif len(key.shape) == 2:  # [batch_size, hidden_size]
            # Take mean across batch dimension
            key = key.mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0)  # [hidden_size]
            importance = importance.mean()  # scalar
        
        # Ensure final shapes are correct
        if key.numel() != self.hidden_size:
            # If tensor is too large, take mean to reduce to hidden_size
            key = key.mean(dim=0) if len(key.shape) > 1 else key
            value = value.mean(dim=0) if len(value.shape) > 1 else value
        
        # Reshape to [hidden_size]
        key = key.view(self.hidden_size)
        value = value.view(self.hidden_size)
        importance = importance.view(1)
        
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
        
        # Ensure tensors are on the correct device
        key = key.to(cache.keys.device)
        value = value.to(cache.values.device)
        importance = importance.to(cache.importance.device)
        
        # Reshape tensors if needed
        if len(key.shape) == 3:  # [batch_size, seq_len, hidden_size]
            # Take mean across batch and sequence dimensions
            key = key.mean(dim=0).mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0).mean(dim=0)  # [hidden_size]
            importance = importance.mean()  # scalar
        elif len(key.shape) == 2:  # [batch_size, hidden_size]
            # Take mean across batch dimension
            key = key.mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0)  # [hidden_size]
            importance = importance.mean()  # scalar
        
        # Ensure final shapes are correct
        if key.numel() != self.hidden_size:
            # If tensor is too large, take mean to reduce to hidden_size
            key = key.mean(dim=0) if len(key.shape) > 1 else key
            value = value.mean(dim=0) if len(value.shape) > 1 else value
        
        # Reshape to [hidden_size]
        key = key.view(self.hidden_size)
        value = value.view(self.hidden_size)
        importance = importance.view(1)
        
        # Update cache with new key-value pair
        cache.update(ptr, key, value, importance)
        
        # Update pointer
        self.cache_ptrs[expert_idx] = (ptr + 1) % cache.size
    
    def forward(self, x):
        # Ensure input is float32
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        
        # Get input shape
        if len(x.shape) == 4:  # [batch_size, num_heads, seq_len, head_dim]
            batch_size, num_heads, seq_len, head_dim = x.shape
            # Reshape to [batch_size, seq_len, hidden_size]
            hidden_size = num_heads * head_dim
            x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
        elif len(x.shape) == 2:  # [batch_size, hidden_size]
            batch_size, hidden_size = x.shape
            x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
            seq_len = 1
        elif len(x.shape) == 3:  # [batch_size, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Calculate routing weights using mean across sequence length
        x_mean = x.mean(dim=1)  # [batch_size, hidden_size]
        routing_weights = self.router(x_mean)  # [batch_size, num_experts]
        
        # Initialize output tensor
        expert_output = torch.zeros(batch_size, seq_len, hidden_size, device=x.device)
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Get expert output
            expert_output_i = expert(x)  # [batch_size, seq_len, hidden_size]
            
            # Get cached values
            cached_values = self.kv_caches[i].get_all()
            
            # Combine with cached values
            if cached_values is not None:
                expert_output_i = expert_output_i + cached_values.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            
            # Update cache with new values
            self.update_cache(i, x.detach(), expert_output_i.detach(), routing_weights[:, i].detach())
            
            # Add to final output weighted by routing probabilities
            expert_output += expert_output_i * routing_weights[:, i].unsqueeze(1).unsqueeze(-1)
        
        # Reshape back to original dimensions if needed
        if len(x.shape) == 4:
            expert_output = expert_output.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        
        return expert_output 