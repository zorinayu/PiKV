import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config
import math

class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])

    def forward(self, x):
        return F.relu(self.dense(x))

class KVCache(nn.Module):
    """
    KV Cache implementation with compression and streaming support.
    """
    def __init__(self, size):
        super(KVCache, self).__init__()
        self.size = size
        self.hidden_size = config['hidden_size']
        
        # Initialize tensors
        self.register_buffer('keys', torch.zeros(size, self.hidden_size))
        self.register_buffer('values', torch.zeros(size, self.hidden_size))
        self.register_buffer('importance', torch.zeros(size))
        self.initialized = True
    
    def update(self, idx, key, value, importance):
        # Update cache at the specified index
        self.keys[idx] = key.mean(dim=0)  # Average across batch
        self.values[idx] = value.mean(dim=0)  # Average across batch
        self.importance[idx] = importance.mean().item()
    
    def get_all(self):
        return self.values.mean(dim=0)  # Return average of all cached values
    
    def set_all(self, data):
        if data is not None:
            self.values.copy_(data.unsqueeze(0).expand(self.size, -1))

class ExternalMemoryCache(nn.Module):
    """
    External memory cache using CXL-based memory disaggregation.
    """
    def __init__(self):
        super(ExternalMemoryCache, self).__init__()
        self.cache = {}
        self.max_size = config.get('external_cache_size', 1000000)
    
    def get(self, key):
        return self.cache.get(key)
    
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()

class PiKVMoE(nn.Module):
    def __init__(self):
        super(PiKVMoE, self).__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(config['num_experts'])])
        self.gate = nn.Linear(config['hidden_size'], config['num_experts'])
        
        # Query-aware KV cache selection
        self.query_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.key_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
        
        # Cache size allocation for each layer
        self.cache_sizes = self.pyramidal_cache_allocation()
        
        # Initialize KV caches for each expert
        self.kv_caches = nn.ModuleList([
            KVCache(size) for size in self.cache_sizes
        ])
        
        # Cache pointers for each expert
        self.register_buffer('cache_ptrs', torch.zeros(config['num_experts'], dtype=torch.long))
        
        # Compression ratio for dynamic KV cache compression
        self.compression_ratio = 1.0
        
        # Memory expansion flag
        self.use_memory_expansion = config.get('use_memory_expansion', False)
        if self.use_memory_expansion:
            self.external_cache = ExternalMemoryCache()

    def pyramidal_cache_allocation(self):
        """
        Calculate the cache size for each layer using the pyramidal allocation policy.
        """
        C1 = config['kv_cache_size']
        d = config['cache_decrement']
        return [C1 - (i - 1) * d for i in range(1, config['num_layers'] + 1)]
    
    def compute_token_importance(self, query, key):
        """
        Compute importance scores for tokens based on query-key attention.
        """
        # Project query and key
        query = self.query_proj(query)
        key = self.key_proj(key)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(config['hidden_size'])
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Compute importance as sum of attention weights
        importance = attention_probs.sum(dim=1)  # [batch_size, seq_len]
        
        return importance
    
    def update_cache(self, expert_idx, key, value, importance):
        """
        Update KV cache for a specific expert with new key-value pairs.
        """
        cache = self.kv_caches[expert_idx]
        ptr = self.cache_ptrs[expert_idx]
        
        # Apply dynamic compression based on importance
        if importance.mean() > 0.5:  # High importance tokens
            self.compression_ratio = 1.0  # No compression
        else:
            self.compression_ratio = 0.5  # Compress by half
        
        # Update cache with new key-value pair
        cache.update(ptr, key, value, importance)
        
        # Update pointer
        self.cache_ptrs[expert_idx] = (ptr + 1) % cache.size
    
    def forward(self, x, query=None):
        # Calculate gate scores
        gate_scores = self.gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        # If query is provided, compute token importance
        if query is not None:
            importance = self.compute_token_importance(query, x)
        else:
            importance = torch.ones(x.size(0), x.size(1), device=x.device)
        
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
                expert_output_i = expert_output_i + cached_values.detach()  # Detach cached values
            
            # Update cache with new values
            self.update_cache(i, x.detach(), expert_output_i.detach(), importance.detach())  # Detach inputs to cache
            
            # Add to final output weighted by gate probabilities
            expert_output += expert_output_i * gate_probs[:, i].unsqueeze(-1)
        
        return expert_output
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint with KV caches.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'cache_ptrs': self.cache_ptrs,
            'kv_caches': [cache.get_all() for cache in self.kv_caches]
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint with KV caches.
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.cache_ptrs.copy_(checkpoint['cache_ptrs'])
        for i, cache_data in enumerate(checkpoint['kv_caches']):
            if cache_data is not None:
                self.kv_caches[i].set_all(cache_data)
