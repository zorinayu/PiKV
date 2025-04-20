import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config
import math
from .shared import ExternalMemoryCache

class LoRALayer(nn.Module):
    """
    LoRA layer implementation for efficient fine-tuning.
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Apply LoRA transformation
        return self.dropout(x @ self.lora_A @ self.lora_B) * self.scaling

class LoRAExpert(nn.Module):
    """
    Expert with LoRA adaptation.
    """
    def __init__(self, hidden_size, rank=4, alpha=1.0):
        super(LoRAExpert, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.lora = LoRALayer(hidden_size, hidden_size, rank=rank, alpha=alpha)
        
    def forward(self, x):
        # Combine base model and LoRA adaptation
        return F.relu(self.dense(x) + self.lora(x))

class LoRAKVCache(nn.Module):
    """
    KV Cache with LoRA adaptation for efficient fine-tuning.
    """
    def __init__(self, size, hidden_size, rank=4, alpha=1.0):
        super(LoRAKVCache, self).__init__()
        self.size = size
        self.hidden_size = hidden_size
        
        # Initialize tensors
        self.register_buffer('keys', torch.zeros(size, hidden_size))
        self.register_buffer('values', torch.zeros(size, hidden_size))
        self.register_buffer('importance', torch.zeros(size))
        
        # LoRA adaptation for keys and values
        self.key_lora = LoRALayer(hidden_size, hidden_size, rank=rank, alpha=alpha)
        self.value_lora = LoRALayer(hidden_size, hidden_size, rank=rank, alpha=alpha)
    
    def update(self, idx, key, value, importance):
        # Update cache at the specified index with LoRA adaptation
        self.keys[idx] = key.mean(dim=0)  # Average across batch
        self.values[idx] = value.mean(dim=0)  # Average across batch
        self.importance[idx] = importance.mean().item()
    
    def get_all(self):
        # Apply LoRA adaptation to cached values
        base_values = self.values.mean(dim=0)
        lora_values = self.value_lora(base_values.unsqueeze(0)).squeeze(0)
        return base_values + lora_values
    
    def set_all(self, data):
        if data is not None:
            self.values.copy_(data.unsqueeze(0).expand(self.size, -1))

class LoRAPiKVMoE(nn.Module):
    """
    PiKV MoE with LoRA adaptation for efficient fine-tuning.
    """
    def __init__(self, rank=4, alpha=1.0):
        super(LoRAPiKVMoE, self).__init__()
        self.experts = nn.ModuleList([LoRAExpert(config['hidden_size'], rank=rank, alpha=alpha) 
                                     for _ in range(config['num_experts'])])
        self.gate = nn.Linear(config['hidden_size'], config['num_experts'])
        
        # Query-aware KV cache selection with LoRA
        self.query_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.key_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.query_lora = LoRALayer(config['hidden_size'], config['hidden_size'], rank=rank, alpha=alpha)
        self.key_lora = LoRALayer(config['hidden_size'], config['hidden_size'], rank=rank, alpha=alpha)
        
        # Cache size allocation for each layer
        self.cache_sizes = self.pyramidal_cache_allocation()
        
        # Initialize KV caches with LoRA for each expert
        self.kv_caches = nn.ModuleList([
            LoRAKVCache(size, config['hidden_size'], rank=rank, alpha=alpha) 
            for size in self.cache_sizes
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
        Compute importance scores for tokens based on query-key attention with LoRA adaptation.
        """
        # Project query and key with LoRA
        query_base = self.query_proj(query)
        key_base = self.key_proj(key)
        
        query_lora = self.query_lora(query)
        key_lora = self.key_lora(key)
        
        query = query_base + query_lora
        key = key_base + key_lora
        
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
            # Get expert output with LoRA adaptation
            expert_output_i = expert(x)
            
            # Get cached values with LoRA adaptation
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
        Save model checkpoint with KV caches and LoRA parameters.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'cache_ptrs': self.cache_ptrs,
            'kv_caches': [cache.get_all() for cache in self.kv_caches]
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint with KV caches and LoRA parameters.
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.cache_ptrs.copy_(checkpoint['cache_ptrs'])
        for i, cache_data in enumerate(checkpoint['kv_caches']):
            if cache_data is not None:
                self.kv_caches[i].set_all(cache_data) 