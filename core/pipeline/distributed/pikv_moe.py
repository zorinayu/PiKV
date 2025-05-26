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
        self.register_buffer('keys', torch.zeros(0, hidden_size))
        self.register_buffer('values', torch.zeros(0, hidden_size))
        self.register_buffer('importance', torch.zeros(0))
    
    def update(self, k: torch.Tensor, v: torch.Tensor, importance: torch.Tensor) -> None:
        # Reshape tensors to match cache dimensions
        batch_size, seq_len, hidden_size = k.shape
        
        # Flatten batch and sequence dimensions
        k_flat = k.reshape(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        v_flat = v.reshape(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        importance_flat = importance.reshape(-1)  # [batch_size * seq_len]
        
        # Update cache
        self.keys = torch.cat([self.keys, k_flat], dim=0)
        self.values = torch.cat([self.values, v_flat], dim=0)
        self.importance = torch.cat([self.importance, importance_flat], dim=0)
        
        # Synchronize across GPUs (optional, can be expensive)
        # Only sync if we need global cache consistency
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(self.keys, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.values, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.importance, op=dist.ReduceOp.SUM)
    
    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.keys, self.values
    
    def clear(self) -> None:
        self.keys = torch.zeros(0, self.hidden_size, device=self.keys.device)
        self.values = torch.zeros(0, self.hidden_size, device=self.values.device)
        self.importance = torch.zeros(0, device=self.importance.device)

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
        batch_size, seq_len, hidden_size = x.shape
        
        q = self.q_proj(x)  # [batch_size, seq_len, hidden_size]
        k = self.k_proj(x)  # [batch_size, seq_len, hidden_size]
        v = self.v_proj(x)  # [batch_size, seq_len, hidden_size]
        
        # Update KV cache if enabled
        if use_cache:
            # Create importance scores (simplified)
            importance = torch.ones_like(k[:, :, 0])  # [batch_size, seq_len]
            self.kv_cache.update(k, v, importance)
            
            # Get cached values (for now, just use current k, v)
            # In a full implementation, you'd retrieve and use cached values
            cached_k, cached_v = self.kv_cache.get_all()
            
            # For simplicity, just use current k, v for attention
            # In practice, you'd combine current and cached values
            k_for_attn = k
            v_for_attn = v
        else:
            k_for_attn = k
            v_for_attn = v
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.hidden_size // self.num_heads)
        k_for_attn = k_for_attn.view(batch_size, seq_len, self.num_heads, self.hidden_size // self.num_heads)
        v_for_attn = v_for_attn.view(batch_size, seq_len, self.num_heads, self.hidden_size // self.num_heads)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k_for_attn = k_for_attn.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v_for_attn = v_for_attn.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        head_dim = self.hidden_size // self.num_heads
        scores = torch.matmul(q, k_for_attn.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v_for_attn)
        
        # Transpose back and reshape
        out = out.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        out = out.view(batch_size, seq_len, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        
        # Apply output projection
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
        # AdaptiveRouter returns 5 values: routing_probs, top_k_indices, weights, load_balancing_loss, importance
        routing_result = self.router(x)
        
        if len(routing_result) == 5:
            # AdaptiveRouter returns 5 values
            routing_probs, top_k_indices, weights, load_balancing_loss, importance = routing_result
        elif len(routing_result) == 4:
            # Other routers return 4 values
            routing_probs, top_k_indices, weights, load_balancing_loss = routing_result
        else:
            raise ValueError(f"Unexpected number of return values from router: {len(routing_result)}")
        
        # Initialize output
        out = torch.zeros_like(x)
        
        # Process through top-k experts
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, :, i]  # [batch_size, seq_len]
            expert_weights = weights[:, :, i]    # [batch_size, seq_len]
            
            # Process through selected experts
            expert_out = torch.zeros_like(x)
            for j in range(self.num_experts):
                # Create mask for this expert
                expert_mask = (expert_idx == j).float()  # [batch_size, seq_len]
                
                if expert_mask.sum() > 0:  # Only process if this expert is selected
                    # Get expert output
                    expert_j_out = self.experts[j](x, use_cache)
                    
                    # Apply expert mask and weights
                    expert_mask_expanded = expert_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
                    expert_weights_expanded = expert_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
                    
                    expert_out += expert_j_out * expert_mask_expanded * expert_weights_expanded
            
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