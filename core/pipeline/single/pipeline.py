import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from .prefill import PrefillStage
from .train import TrainStage
from .inference import InferenceStage

class Pipeline(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, kv_cache_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_cache_size = kv_cache_size
        
        # Initialize stages
        self.prefill = PrefillStage(hidden_size, num_heads, kv_cache_size)
        self.train = TrainStage(hidden_size, num_heads, kv_cache_size)
        self.inference = InferenceStage(hidden_size, num_heads, kv_cache_size)
        
        # Initialize input embedding
        self.embedding = nn.Embedding(50257, hidden_size)  # GPT-2 vocabulary size
        
        # Initialize output projection
        self.output_proj = nn.Linear(hidden_size, 50257)
        
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                mode: str = 'train') -> Dict[str, torch.Tensor]:
        """
        Forward pass through the pipeline.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: Optional [batch_size, seq_len]
            past_key_values: Optional tuple of (past_key, past_value)
            mode: 'train' or 'inference'
            
        Returns:
            Dictionary containing:
                - logits: [batch_size, seq_len, vocab_size]
                - past_key_values: Tuple of (new_key, new_value) if mode='inference'
        """
        # Get input embeddings
        hidden_states = self.embedding(input_ids)
        
        if mode == 'train':
            # Training mode
            output = self.train(hidden_states, attention_mask)
            logits = self.output_proj(output)
            return {'logits': logits}
            
        elif mode == 'inference':
            # Inference mode
            output, new_key_values = self.inference(
                hidden_states,
                attention_mask,
                past_key_values
            )
            logits = self.output_proj(output)
            return {
                'logits': logits,
                'past_key_values': new_key_values
            }
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def clear_cache(self) -> None:
        """Clear all KV caches."""
        self.prefill.clear_cache()
        self.train.clear_cache()
        self.inference.clear_cache() 