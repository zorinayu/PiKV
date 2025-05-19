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

class PyramidCompressor(BaseCompressor):
    """
    Pyramid cache compressor that implements hierarchical compression
    Inspired by PyramidKV which uses a pyramid structure to compress KV cache
    """
    def __init__(
        self, 
        hidden_size: int, 
        compression_ratio: float = 0.5,
        num_levels: int = 3,
        decay_factor: float = 0.8
    ):
        super(PyramidCompressor, self).__init__(hidden_size)
        self.compression_ratio = compression_ratio
        self.num_levels = num_levels
        self.decay_factor = decay_factor
        
        # Initialize pyramid compression layers
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        # Create pyramid of encoding/decoding layers
        current_size = hidden_size
        for i in range(num_levels):
            # Calculate compressed size for this level
            output_size = max(int(current_size * compression_ratio), 1)
            
            # Create encoder and decoder for this level
            self.encoder_layers.append(nn.Linear(current_size, output_size))
            self.decoder_layers.append(nn.Linear(output_size, current_size))
            
            # Update size for next level
            current_size = output_size
            
            # Adjust compression ratio for next level
            compression_ratio *= decay_factor
        
        # Initialize weights
        self._init_weights()
        
        # Cache for statistics tracking
        self.register_buffer('compression_stats', torch.zeros(4))  # [Total, Avg, Max, Min]
        self.register_buffer('sample_count', torch.tensor(0.0))
    
    def _init_weights(self):
        """Initialize weights for all layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _compress_level(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Apply compression at a specific pyramid level"""
        if level >= len(self.encoder_layers):
            return x
        
        # Apply encoder
        x = self.encoder_layers[level](x)
        
        # Apply activation
        x = F.relu(x)
        
        return x
    
    def _decompress_level(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Apply decompression at a specific pyramid level"""
        if level >= len(self.decoder_layers):
            return x
        
        # Apply decoder
        x = self.decoder_layers[level](x)
        
        # Apply activation
        x = F.relu(x)
        
        return x
    
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply pyramid compression to KV cache
        
        Args:
            keys: Key tensors [batch_size, seq_len, hidden_size]
            values: Value tensors [batch_size, seq_len, hidden_size]
            importance: Optional importance scores [batch_size, seq_len]
            
        Returns:
            compressed_keys: Compressed key tensors
            compressed_values: Compressed value tensors
        """
        batch_size, seq_len, hidden_size = keys.shape
        
        # If importance scores are provided, apply importance-weighted compression
        if importance is not None:
            # Normalize importance scores
            if importance.dim() == 2:  # [batch_size, seq_len]
                importance = importance.unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            importance = importance / (importance.max() + 1e-5)
            
            # Sort by importance
            importance_flat = importance.reshape(-1)
            sorted_indices = torch.argsort(importance_flat, descending=True)
            
            # Distribute across compression levels
            level_sizes = []
            remaining = sorted_indices.size(0)
            
            for i in range(self.num_levels):
                # Last level gets all remaining items
                if i == self.num_levels - 1:
                    level_sizes.append(remaining)
                else:
                    # More important tokens get less compression
                    level_size = int(remaining * (1 - self.decay_factor) * (self.decay_factor ** i))
                    level_sizes.append(level_size)
                    remaining -= level_size
            
            # Flatten for processing
            keys_flat = keys.reshape(-1, hidden_size)
            values_flat = values.reshape(-1, hidden_size)
            
            # Initialize compressed tensors
            compressed_keys = torch.zeros_like(keys_flat)
            compressed_values = torch.zeros_like(values_flat)
            
            # Apply appropriate compression level to each group
            start_idx = 0
            for level, size in enumerate(level_sizes):
                if size <= 0:
                    continue
                
                end_idx = start_idx + size
                level_indices = sorted_indices[start_idx:end_idx]
                
                # Select tokens for this level
                level_keys = keys_flat[level_indices]
                level_values = values_flat[level_indices]
                
                # Apply appropriate level of compression based on importance
                compressed_level_keys = level_keys
                compressed_level_values = level_values
                
                # Apply compression through pyramid levels
                for i in range(level):
                    compressed_level_keys = self._compress_level(compressed_level_keys, i)
                    compressed_level_values = self._compress_level(compressed_level_values, i)
                
                # Apply decompression through pyramid levels
                for i in range(level-1, -1, -1):
                    compressed_level_keys = self._decompress_level(compressed_level_keys, i)
                    compressed_level_values = self._decompress_level(compressed_level_values, i)
                
                # Place back in original order
                compressed_keys[level_indices] = compressed_level_keys
                compressed_values[level_indices] = compressed_level_values
                
                start_idx = end_idx
            
            # Reshape back to original dimensions
            compressed_keys = compressed_keys.reshape(batch_size, seq_len, hidden_size)
            compressed_values = compressed_values.reshape(batch_size, seq_len, hidden_size)
        
        else:
            # If no importance scores, apply uniform compression to all tokens
            compressed_keys = keys
            compressed_values = values
            
            # Apply compression through all pyramid levels
            for i in range(self.num_levels):
                compressed_keys = self._compress_level(compressed_keys, i)
                compressed_values = self._compress_level(compressed_values, i)
            
            # Apply decompression through all pyramid levels
            for i in range(self.num_levels-1, -1, -1):
                compressed_keys = self._decompress_level(compressed_keys, i)
                compressed_values = self._decompress_level(compressed_values, i)
        
        # Update compression statistics
        with torch.no_grad():
            # Calculate compression ratio (compressed size / original size)
            current_ratio = compressed_keys.numel() / keys.numel()
            
            # Update statistics
            self.compression_stats[0] += current_ratio  # Total ratio
            self.compression_stats[1] = self.compression_stats[0] / (self.sample_count + 1)  # Average ratio
            
            if self.compression_stats[2] == 0:
                self.compression_stats[2] = current_ratio  # Initial max
            else:
                self.compression_stats[2] = max(self.compression_stats[2], current_ratio)  # Max ratio
                
            if self.compression_stats[3] == 0:
                self.compression_stats[3] = current_ratio  # Initial min
            else:
                self.compression_stats[3] = min(self.compression_stats[3], current_ratio)  # Min ratio
                
            self.sample_count += 1
        
        return compressed_keys, compressed_values
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        stats = super().get_compression_stats()
        
        # Add pyramid-specific stats
        stats.update({
            "compression_ratio": self.compression_stats[1].item(),  # Average ratio
            "max_compression_ratio": self.compression_stats[2].item(),
            "min_compression_ratio": self.compression_stats[3].item(),
            "num_levels": self.num_levels,
            "decay_factor": self.decay_factor,
            "memory_reduction": 1.0 - self.compression_stats[1].item(),
            "sample_count": self.sample_count.item()
        })
        
        return stats

class FastVCompressor(BaseCompressor):
    """
    FastV Compressor inspired by FastV which accelerates value cache access
    by using value-specific optimizations:
    1. Value grouping for similar values
    2. Centroid-based compression
    3. Sparse updates for frequently accessed values
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_centroids: int = 32,
        sparsity_threshold: float = 0.1,
        update_interval: int = 100
    ):
        super(FastVCompressor, self).__init__(hidden_size)
        self.num_centroids = num_centroids
        self.sparsity_threshold = sparsity_threshold
        self.update_interval = update_interval
        
        # Initialize centroids
        self.register_buffer('key_centroids', torch.zeros(num_centroids, hidden_size))
        self.register_buffer('value_centroids', torch.zeros(num_centroids, hidden_size))
        
        # Centroid usage counts
        self.register_buffer('centroid_counts', torch.zeros(num_centroids))
        
        # Access frequency for sparse updates
        self.register_buffer('access_frequency', torch.zeros(num_centroids))
        
        # Center projection for assignments
        self.center_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_centroids)
        )
        
        # Trainable transformation for reconstruction
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Iteration counter
        self.register_buffer('iterations', torch.tensor(0))
        
        # Compression statistics
        self.register_buffer('compression_ratio', torch.tensor(float(num_centroids) / (hidden_size * 100)))
        self.register_buffer('sample_count', torch.tensor(0.0))
        
        # Initialize all weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _assign_centroids(self, x: torch.Tensor) -> torch.Tensor:
        """Assign each vector to nearest centroid"""
        # Flatten input to [N, hidden_size] where N = batch_size * seq_len
        x_flat = x.reshape(-1, self.hidden_size)
        
        # Project to get centroid scores [N, num_centroids]
        centroid_scores = self.center_proj(x_flat)
        
        # Get centroid assignments using argmax
        centroid_indices = torch.argmax(centroid_scores, dim=-1)
        
        # Update access frequency
        with torch.no_grad():
            for idx in centroid_indices:
                self.access_frequency[idx] += 1
        
        return centroid_indices
    
    def _update_centroids(self, keys: torch.Tensor, values: torch.Tensor, centroid_indices: torch.Tensor):
        """Update centroids based on assigned vectors"""
        # Only update periodically
        if self.iterations % self.update_interval != 0:
            return
        
        # Flatten inputs
        keys_flat = keys.reshape(-1, self.hidden_size)
        values_flat = values.reshape(-1, self.hidden_size)
        
        with torch.no_grad():
            # Reset centroids and counts
            self.key_centroids.zero_()
            self.value_centroids.zero_()
            self.centroid_counts.zero_()
            
            # Sum vectors for each centroid
            for i, idx in enumerate(centroid_indices):
                self.key_centroids[idx] += keys_flat[i]
                self.value_centroids[idx] += values_flat[i]
                self.centroid_counts[idx] += 1
            
            # Calculate means for non-empty centroids
            mask = self.centroid_counts > 0
            self.key_centroids[mask] /= self.centroid_counts[mask].unsqueeze(1)
            self.value_centroids[mask] /= self.centroid_counts[mask].unsqueeze(1)
            
            # Initialize empty centroids randomly if any
            empty_mask = self.centroid_counts == 0
            if empty_mask.any():
                # Find most frequently accessed centroids
                _, most_frequent = torch.topk(self.access_frequency, min(3, self.num_centroids))
                
                # For each empty centroid, initialize with a perturbed version of a popular one
                for idx in torch.nonzero(empty_mask, as_tuple=True)[0]:
                    source_idx = most_frequent[torch.randint(0, len(most_frequent), (1,))]
                    
                    # Add random noise to create a new centroid
                    self.key_centroids[idx] = self.key_centroids[source_idx] + torch.randn_like(self.key_centroids[0]) * 0.1
                    self.value_centroids[idx] = self.value_centroids[source_idx] + torch.randn_like(self.value_centroids[0]) * 0.1
    
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply FastV compression to KV cache
        
        Args:
            keys: Key tensors [batch_size, seq_len, hidden_size]
            values: Value tensors [batch_size, seq_len, hidden_size]
            importance: Optional importance scores [batch_size, seq_len]
            
        Returns:
            compressed_keys: Compressed key tensors
            compressed_values: Compressed value tensors
        """
        batch_size, seq_len, hidden_size = keys.shape
        
        # Assign each vector to nearest centroid
        centroid_indices = self._assign_centroids(keys)
        
        # Update centroids if needed
        self._update_centroids(keys, values, centroid_indices)
        
        # Get centroids for each token
        compressed_keys_flat = self.key_centroids[centroid_indices]
        compressed_values_flat = self.value_centroids[centroid_indices]
        
        # Apply trainable transformation
        transformed_keys = self.transform(compressed_keys_flat)
        transformed_values = self.transform(compressed_values_flat)
        
        # Reshape back to original dimensions
        compressed_keys = transformed_keys.reshape(batch_size, seq_len, hidden_size)
        compressed_values = transformed_values.reshape(batch_size, seq_len, hidden_size)
        
        # For important tokens, add residual from original
        if importance is not None:
            # Threshold importance to create a sparse mask
            if importance.dim() == 2:
                importance = importance.unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Create sparse mask for important tokens
            sparse_mask = (importance > self.sparsity_threshold).float()
            
            # Apply residual only for important tokens
            # This maintains high fidelity for important tokens while compressing others
            compressed_keys = compressed_keys + sparse_mask * (keys - compressed_keys)
            compressed_values = compressed_values + sparse_mask * (values - compressed_values)
        
        # Update statistics
        with torch.no_grad():
            self.iterations += 1
            self.sample_count += 1
        
        return compressed_keys, compressed_values
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        stats = super().get_compression_stats()
        
        # Calculate actual compression ratio based on centroids vs full cache
        actual_ratio = float(self.num_centroids) / (self.hidden_size * 100)
        
        # Get most used centroids
        with torch.no_grad():
            most_used, _ = torch.topk(self.centroid_counts, min(5, self.num_centroids))
            utilization = torch.sum(self.centroid_counts > 0).item() / self.num_centroids
        
        stats.update({
            "compression_ratio": actual_ratio,
            "memory_reduction": 1.0 - actual_ratio,
            "num_centroids": self.num_centroids,
            "active_centroids": torch.sum(self.centroid_counts > 0).item(),
            "centroid_utilization": utilization,
            "iterations": self.iterations.item(),
            "sample_count": self.sample_count.item()
        })
        
        return stats 