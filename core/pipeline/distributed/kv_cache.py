import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple
from ..single.kv_cache_compression import KVCacheCompressor

class DistributedKVCache(nn.Module):
    def __init__(self, size: int, hidden_size: int, expert_id: int, world_size: int):
        super().__init__()
        self.size = size
        self.hidden_size = hidden_size
        self.expert_id = expert_id
        self.world_size = world_size
        
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
    
    def update(self, idx: int, key: torch.Tensor, value: torch.Tensor, importance: torch.Tensor) -> None:
        # Reshape input if needed
        if len(key.shape) == 3:  # [batch_size, seq_len, hidden_size]
            key = key.mean(dim=0).mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0).mean(dim=0)  # [hidden_size]
        elif len(key.shape) == 2:  # [seq_len, hidden_size]
            key = key.mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0)  # [hidden_size]
        
        # Update cache
        self.keys[idx] = key
        self.values[idx] = value
        self.importance[idx] = importance.mean().item()
        
        # Synchronize across GPUs
        if dist.is_initialized():
            dist.all_reduce(self.keys, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.values, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.importance, op=dist.ReduceOp.SUM)
            
            # Average the values
            self.keys /= self.world_size
            self.values /= self.world_size
            self.importance /= self.world_size
    
    def get_all(self) -> Optional[torch.Tensor]:
        # Apply compression to cached values
        compressed_keys, compressed_values = self.compressor(
            self.keys.unsqueeze(0),
            self.values.unsqueeze(0),
            self.importance.unsqueeze(0)
        )
        
        # Synchronize compressed values across GPUs
        if dist.is_initialized():
            dist.all_reduce(compressed_values, op=dist.ReduceOp.SUM)
            compressed_values /= self.world_size
        
        return compressed_values.squeeze(0)
    
    def set_all(self, data: Optional[torch.Tensor]) -> None:
        if data is not None:
            self.values.copy_(data.unsqueeze(0).expand(self.size, -1))
            
            # Synchronize across GPUs
            if dist.is_initialized():
                dist.all_reduce(self.values, op=dist.ReduceOp.SUM)
                self.values /= self.world_size 