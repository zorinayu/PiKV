import torch
import torch.nn as nn
from typing import Optional, Tuple
from .prefill import PrefillStage

class TrainStage(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, kv_cache_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Initialize prefill stage
        self.prefill = PrefillStage(hidden_size, num_heads, kv_cache_size)
        
        # Initialize attention layers
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with KV cache management.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional [batch_size, seq_len]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project queries, keys, and values
        q = self.q_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        k = self.k_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        v = self.v_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Update KV cache
        self.prefill.update_cache(k, v, attention_mask)
        
        # Get cached keys and values
        cached_k, cached_v, cache_mask = self.prefill.get_cache()
        
        # Compute attention scores
        scores = torch.matmul(q, cached_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, cached_v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        # Add residual connection and layer normalization
        output = self.layer_norm(hidden_states + output)
        
        return output
    
    def clear_cache(self) -> None:
        """Clear the KV cache."""
        self.prefill.clear_cache() 