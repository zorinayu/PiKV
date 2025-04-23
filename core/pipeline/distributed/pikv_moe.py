import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple, List
from ..single.routing_strategy import AdaptiveRouter
from ..single.kv_cache_compression import KVCacheCompressor

class DistributedKVCache(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Initialize cache buffers
        self.register_buffer('keys', torch.zeros(0, num_heads, self.head_dim))
        self.register_buffer('values', torch.zeros(0, num_heads, self.head_dim))
        self.register_buffer('importance', torch.zeros(0, num_heads))
    
    def update(self, k: torch.Tensor, v: torch.Tensor, importance: torch.Tensor) -> None:
        # Update cache
        self.keys = torch.cat([self.keys, k], dim=0)
        self.values = torch.cat([self.values, v], dim=0)
        self.importance = torch.cat([self.importance, importance], dim=0)
        
        # Synchronize across GPUs
        dist.all_reduce(self.keys, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.values, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.importance, op=dist.ReduceOp.SUM)
    
    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.keys, self.values
    
    def clear(self) -> None:
        self.keys = torch.zeros(0, self.num_heads, self.head_dim, device=self.keys.device)
        self.values = torch.zeros(0, self.num_heads, self.head_dim, device=self.values.device)
        self.importance = torch.zeros(0, self.num_heads, device=self.importance.device)

class DistributedExpert(nn.Module):
    def __init__(self, expert_id: int, world_size: int, hidden_size: int, expert_size: int, num_heads: int):
        super().__init__()
        self.expert_id = expert_id
        self.world_size = world_size
        self.hidden_size = hidden_size
        self.expert_size = expert_size
        self.num_heads = num_heads
        
        # Initialize expert layers
        self.fc1 = nn.Linear(hidden_size, expert_size)
        self.fc2 = nn.Linear(expert_size, hidden_size)
        
        # Initialize attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Initialize KV cache
        self.kv_cache = DistributedKVCache(hidden_size, num_heads)
    
    def forward(self, x: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        # Process through expert layers
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        
        # Compute attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Update KV cache if enabled
        if use_cache:
            self.kv_cache.update(k, v, torch.ones_like(k))
            k, v = self.kv_cache.get_all()
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = self.o_proj(out)
        
        return out + h

class DistributedPiKVMoE(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, expert_size: int, 
                 num_heads: int, top_k: int = 2, compression_ratio: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_size = expert_size
        self.num_heads = num_heads
        self.top_k = top_k
        
        # Initialize experts
        self.experts = nn.ModuleList([
            DistributedExpert(
                i,
                dist.get_world_size(),
                hidden_size,
                expert_size,
                num_heads
            )
            for i in range(num_experts)
        ])
        
        # Initialize router
        self.router = AdaptiveRouter(hidden_size, num_experts, top_k)
        
        # Initialize KV cache compressor
        self.compressor = KVCacheCompressor(hidden_size, compression_ratio)
        
        # Initialize distributed parameters
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
    
    def forward(self, x: torch.Tensor, use_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get routing probabilities and top-k experts
        routing_probs, top_k_indices, weights, load_balancing_loss = self.router(x)
        
        # Initialize output
        out = torch.zeros_like(x)
        
        # Process through top-k experts
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_mask = torch.zeros(self.num_experts, device=x.device)
            expert_mask.scatter_(0, expert_idx, weights[:, i])
            
            # Process through selected experts
            expert_out = torch.zeros_like(x)
            for j in range(self.num_experts):
                if expert_mask[j] > 0:
                    expert_out += expert_mask[j] * self.experts[j](x, use_cache)
            
            out += expert_out
        
        # Synchronize across GPUs
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        
        return out, load_balancing_loss
    
    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'cache_ptrs': self.cache_ptrs,
            'kv_caches': [cache.get_all() for cache in self.kv_caches]
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.cache_ptrs.copy_(checkpoint['cache_ptrs'])
        for i, cache_data in enumerate(checkpoint['kv_caches']):
            if cache_data is not None:
                self.kv_caches[i].set_all(cache_data) 