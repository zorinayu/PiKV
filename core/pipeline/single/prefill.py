import torch
import torch.nn as nn
from typing import Optional, Tuple

class PrefillStage(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, kv_cache_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.kv_cache_size = kv_cache_size
        
        # Initialize KV cache
        self.register_buffer('key_cache', torch.zeros(kv_cache_size, num_heads, self.head_dim))
        self.register_buffer('value_cache', torch.zeros(kv_cache_size, num_heads, self.head_dim))
        self.register_buffer('cache_mask', torch.zeros(kv_cache_size, dtype=torch.bool))
        
        # Cache pointer
        self.cache_ptr = 0
        
    def update_cache(self, 
                    key: torch.Tensor, 
                    value: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> None:
        """
        Update KV cache with new key-value pairs.
        
        Args:
            key: [batch_size, num_heads, seq_len, head_dim]
            value: [batch_size, num_heads, seq_len, head_dim]
            mask: Optional [batch_size, seq_len] attention mask
        """
        batch_size, num_heads, seq_len, head_dim = key.shape
        
        # Ensure tensors are on correct device
        key = key.to(self.key_cache.device)
        value = value.to(self.value_cache.device)
        
        # Update cache
        start_idx = self.cache_ptr
        end_idx = start_idx + seq_len
        
        if end_idx > self.kv_cache_size:
            # Handle cache overflow
            remaining = self.kv_cache_size - start_idx
            self.key_cache[start_idx:] = key[0, :, :remaining].detach()
            self.value_cache[start_idx:] = value[0, :, :remaining].detach()
            self.cache_ptr = 0
            start_idx = 0
            end_idx = seq_len - remaining
        
        self.key_cache[start_idx:end_idx] = key[0, :, :end_idx-start_idx].detach()
        self.value_cache[start_idx:end_idx] = value[0, :, :end_idx-start_idx].detach()
        
        if mask is not None:
            self.cache_mask[start_idx:end_idx] = mask[0, :end_idx-start_idx].detach()
        
        self.cache_ptr = end_idx % self.kv_cache_size
    
    def get_cache(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get current KV cache state.
        
        Returns:
            key_cache: [cache_size, num_heads, head_dim]
            value_cache: [cache_size, num_heads, head_dim]
            cache_mask: [cache_size]
        """
        return self.key_cache, self.value_cache, self.cache_mask
    
    def clear_cache(self) -> None:
        """Clear the KV cache."""
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.cache_mask.zero_()
        self.cache_ptr = 0 