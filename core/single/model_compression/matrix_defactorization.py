import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List, Optional, Union, Any

class BaseCompressor(nn.Module):
    """
    Base compressor class for KV cache compression
    """
    def __init__(self, hidden_size: int):
        super(BaseCompressor, self).__init__()
        self.hidden_size = hidden_size
        
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Base implementation for compressing KV cache
        
        Args:
            keys: Key tensors [batch_size, seq_len, hidden_size]
            values: Value tensors [batch_size, seq_len, hidden_size]
            importance: Optional importance scores [batch_size, seq_len]
            
        Returns:
            compressed_keys: Compressed key tensors
            compressed_values: Compressed value tensors
        """
        # Base class doesn't implement compression, just returns originals
        return keys, values
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return {
            "hidden_size": self.hidden_size,
        }

class LoRACompressor(BaseCompressor):
    """
    LoRA (Low-Rank Adaptation) based compressor for KV cache
    Reduces parameter count by using low-rank decomposition
    """
    def __init__(
        self, 
        hidden_size: int, 
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super(LoRACompressor, self).__init__(hidden_size)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize LoRA matrices for keys
        self.key_lora_A = nn.Parameter(torch.zeros(hidden_size, rank))
        self.key_lora_B = nn.Parameter(torch.zeros(rank, hidden_size))
        
        # Initialize LoRA matrices for values
        self.value_lora_A = nn.Parameter(torch.zeros(hidden_size, rank))
        self.value_lora_B = nn.Parameter(torch.zeros(rank, hidden_size))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        # Cache for intermediate representations
        self.register_buffer('key_cache', torch.zeros(1, hidden_size))
        self.register_buffer('value_cache', torch.zeros(1, hidden_size))
        
        # Stats tracking
        self.register_buffer('compression_ratio', torch.tensor(float(rank) / hidden_size))
        self.register_buffer('sample_count', torch.tensor(0.0))
    
    def _init_weights(self):
        """Initialize weights for LoRA matrices"""
        # Use kaiming initialization for A matrices
        nn.init.kaiming_uniform_(self.key_lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.value_lora_A, a=math.sqrt(5))
        
        # Initialize B matrices to zero
        nn.init.zeros_(self.key_lora_B)
        nn.init.zeros_(self.value_lora_B)
    
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply LoRA compression to KV cache
        
        Args:
            keys: Key tensors [batch_size, seq_len, hidden_size]
            values: Value tensors [batch_size, seq_len, hidden_size]
            importance: Optional importance scores [batch_size, seq_len]
            
        Returns:
            compressed_keys: Compressed key tensors
            compressed_values: Compressed value tensors
        """
        batch_size, seq_len, hidden_size = keys.shape
        
        # Reshape if needed to work with batched inputs
        keys_flat = keys.reshape(-1, hidden_size)  # [batch_size*seq_len, hidden_size]
        values_flat = values.reshape(-1, hidden_size)  # [batch_size*seq_len, hidden_size]
        
        # Apply LoRA transformations to keys
        key_delta = self.dropout(keys_flat @ self.key_lora_A) @ self.key_lora_B
        key_delta = key_delta * self.scaling
        
        # Apply LoRA transformations to values
        value_delta = self.dropout(values_flat @ self.value_lora_A) @ self.value_lora_B
        value_delta = value_delta * self.scaling
        
        # Combine with original tensors
        compressed_keys = keys_flat + key_delta
        compressed_values = values_flat + value_delta
        
        # Update cache with average representation
        with torch.no_grad():
            self.key_cache = keys_flat.mean(dim=0, keepdim=True)
            self.value_cache = values_flat.mean(dim=0, keepdim=True)
            self.sample_count += 1
        
        # Reshape back to original dimensions
        compressed_keys = compressed_keys.reshape(batch_size, seq_len, hidden_size)
        compressed_values = compressed_values.reshape(batch_size, seq_len, hidden_size)
        
        return compressed_keys, compressed_values
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        stats = super().get_compression_stats()
        stats.update({
            "rank": self.rank,
            "alpha": self.alpha,
            "compression_ratio": self.compression_ratio.item(),
            "memory_reduction": 1.0 - self.compression_ratio.item(),
            "parameter_count": self.rank * self.hidden_size * 4,  # 4 matrices: A and B for keys and values
            "sample_count": self.sample_count.item()
        })
        return stats

class LoRAPlusCompressor(LoRACompressor):
    """
    LoRA+ compressor with additional improvements:
    1. Selective application based on importance
    2. Adaptive rank selection
    3. Residual scaling
    """
    def __init__(
        self,
        hidden_size: int,
        ranks: List[int] = [4, 8, 16],  # Multiple ranks for different importance levels
        alpha: float = 16.0,
        dropout: float = 0.0,
        importance_thresholds: List[float] = [0.3, 0.7]  # Thresholds for rank selection
    ):
        # Initialize with the largest rank first
        super(LoRAPlusCompressor, self).__init__(hidden_size, max(ranks), alpha, dropout)
        
        self.ranks = sorted(ranks)  # Sort ranks in ascending order
        self.importance_thresholds = importance_thresholds
        
        # Create multiple LoRA modules for different ranks
        self.lora_modules = nn.ModuleList()
        for rank in ranks:
            self.lora_modules.append(
                nn.ModuleDict({
                    'key_lora_A': nn.Parameter(torch.zeros(hidden_size, rank)),
                    'key_lora_B': nn.Parameter(torch.zeros(rank, hidden_size)),
                    'value_lora_A': nn.Parameter(torch.zeros(hidden_size, rank)),
                    'value_lora_B': nn.Parameter(torch.zeros(rank, hidden_size))
                })
            )
        
        # Initialize all LoRA weights
        self._init_all_weights()
        
        # Attention projection for importance-based selection
        self.attention_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Register additional buffers for tracking
        self.register_buffer('rank_usage', torch.zeros(len(ranks)))
        self.register_buffer('residual_scale', torch.tensor(1.0))
    
    def _init_all_weights(self):
        """Initialize weights for all LoRA modules"""
        for i, module_dict in enumerate(self.lora_modules):
            # Use kaiming initialization for A matrices
            nn.init.kaiming_uniform_(module_dict['key_lora_A'], a=math.sqrt(5))
            nn.init.kaiming_uniform_(module_dict['value_lora_A'], a=math.sqrt(5))
            
            # Initialize B matrices to zero
            nn.init.zeros_(module_dict['key_lora_B'])
            nn.init.zeros_(module_dict['value_lora_B'])
    
    def _select_rank_index(self, importance: torch.Tensor) -> int:
        """Select rank index based on importance"""
        if importance is None:
            return len(self.ranks) - 1  # Use highest rank if no importance provided
        
        # Average importance
        avg_importance = importance.mean().item()
        
        # Select rank based on importance thresholds
        for i, threshold in enumerate(self.importance_thresholds):
            if avg_importance < threshold:
                return i
        
        # If importance is higher than all thresholds, use the highest rank
        return len(self.ranks) - 1
    
    def _compute_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute importance scores if not provided"""
        # Use attention projection to estimate importance
        x_mean = x.mean(dim=1)  # [batch_size, hidden_size]
        importance = self.attention_proj(x_mean)  # [batch_size, 1]
        return importance
    
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply LoRA+ compression to KV cache with adaptive rank selection
        
        Args:
            keys: Key tensors [batch_size, seq_len, hidden_size]
            values: Value tensors [batch_size, seq_len, hidden_size]
            importance: Optional importance scores [batch_size, seq_len]
            
        Returns:
            compressed_keys: Compressed key tensors
            compressed_values: Compressed value tensors
        """
        batch_size, seq_len, hidden_size = keys.shape
        
        # Compute importance if not provided
        if importance is None:
            importance = self._compute_importance(keys)
        
        # Select rank based on importance
        rank_idx = self._select_rank_index(importance)
        selected_rank = self.ranks[rank_idx]
        selected_module = self.lora_modules[rank_idx]
        
        # Update rank usage stats
        with torch.no_grad():
            self.rank_usage[rank_idx] += 1
        
        # Reshape if needed to work with batched inputs
        keys_flat = keys.reshape(-1, hidden_size)  # [batch_size*seq_len, hidden_size]
        values_flat = values.reshape(-1, hidden_size)  # [batch_size*seq_len, hidden_size]
        
        # Apply selected LoRA transformations to keys
        key_delta = self.dropout(keys_flat @ selected_module['key_lora_A']) @ selected_module['key_lora_B']
        key_delta = key_delta * (self.scaling * self.residual_scale)
        
        # Apply selected LoRA transformations to values
        value_delta = self.dropout(values_flat @ selected_module['value_lora_A']) @ selected_module['value_lora_B']
        value_delta = value_delta * (self.scaling * self.residual_scale)
        
        # Combine with original tensors
        compressed_keys = keys_flat + key_delta
        compressed_values = values_flat + value_delta
        
        # Update cache with average representation
        with torch.no_grad():
            self.key_cache = keys_flat.mean(dim=0, keepdim=True)
            self.value_cache = values_flat.mean(dim=0, keepdim=True)
            self.sample_count += 1
            
            # Adaptively adjust residual scale based on importance
            if importance is not None:
                self.residual_scale = torch.clamp(importance.mean() * 2.0, 0.5, 1.5)
        
        # Reshape back to original dimensions
        compressed_keys = compressed_keys.reshape(batch_size, seq_len, hidden_size)
        compressed_values = compressed_values.reshape(batch_size, seq_len, hidden_size)
        
        return compressed_keys, compressed_values
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        stats = super().get_compression_stats()
        
        # Add LoRA+ specific stats
        rank_usage = {}
        if self.sample_count > 0:
            for i, rank in enumerate(self.ranks):
                rank_usage[f"rank_{rank}"] = (self.rank_usage[i] / self.sample_count).item()
        
        stats.update({
            "ranks": self.ranks,
            "rank_usage": rank_usage,
            "residual_scale": self.residual_scale.item(),
            "adaptive_ranks": True
        })
        
        return stats 