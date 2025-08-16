import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List, Optional, Union, Any

class BaseCompressor(nn.Module):
    """
    Base KV cache compressor with common interface
    Provides basic structure and interface for KV cache compression
    """
    def __init__(self, hidden_size: int, compression_ratio: float = 0.5):
        super(BaseCompressor, self).__init__()
        self.hidden_size = hidden_size
        self.compression_ratio = compression_ratio
        
    def forward(
        self, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV cache - base implementation returns original values
        
        Args:
            keys: Cache key tensor [batch_size, seq_len, hidden_size]
            values: Cache value tensor [batch_size, seq_len, hidden_size]
            importance: Optional importance scores [batch_size, seq_len]
            
        Returns:
            compressed_keys: Compressed key tensor
            compressed_values: Compressed value tensor
        """
        return keys, values
    
    def get_compression_stats(self) -> Dict[str, float]:
        """
        Get compression statistics
        
        Returns:
            stats: Dictionary containing compression ratio, memory reduction, etc.
        """
        return {
            "compression_ratio": self.compression_ratio,
            "memory_reduction": 1.0 - self.compression_ratio
        }

class LoRACompressor(BaseCompressor):
    """LoRA-based compression using low-rank adaptation"""
    
    def __init__(self, hidden_size: int, rank: int = 16, alpha: float = 32.0, dropout: float = 0.1):
        super().__init__(hidden_size, compression_ratio=rank/hidden_size)
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices for keys and values
        self.key_lora_A = nn.Linear(hidden_size, rank, bias=False)
        self.key_lora_B = nn.Linear(rank, hidden_size, bias=False)
        self.value_lora_A = nn.Linear(hidden_size, rank, bias=False)
        self.value_lora_B = nn.Linear(rank, hidden_size, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LoRA weights"""
        for m in [self.key_lora_A, self.key_lora_B, self.value_lora_A, self.value_lora_B]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        
        # Zero initialize B matrices
        nn.init.zeros_(self.key_lora_B.weight)
        nn.init.zeros_(self.value_lora_B.weight)
    
    def forward(self, keys: torch.Tensor, values: torch.Tensor, importance: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress using LoRA"""
        # Apply LoRA compression
        key_compressed = self.key_lora_B(self.dropout(self.key_lora_A(keys)))
        value_compressed = self.value_lora_B(self.dropout(self.value_lora_A(values)))
        
        # Scale by alpha/rank
        scaling = self.alpha / self.rank
        key_compressed = key_compressed * scaling
        value_compressed = value_compressed * scaling
        
        return key_compressed, value_compressed

class PyramidCompressor(BaseCompressor):
    """
    Pyramid compression with hierarchical levels
    Implements hierarchical compression scheme with different compression ratios at different levels
    Reference: PyramidKV
    """
    def __init__(
        self, 
        hidden_size: int, 
        compression_ratio: float = 0.5,
        num_levels: int = 3,
        decay_factor: float = 0.8
    ):
        super().__init__(hidden_size, compression_ratio)
        self.num_levels = num_levels
        self.decay_factor = decay_factor
        
        # Build pyramid layers
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        current_size = hidden_size
        for i in range(num_levels):
            output_size = max(int(current_size * compression_ratio), 1)
            
            self.encoder_layers.append(nn.Linear(current_size, output_size))
            self.decoder_layers.append(nn.Linear(output_size, current_size))
            
            current_size = output_size
            compression_ratio *= decay_factor
        
        self._init_weights()
        
        # Cache statistics
        self.register_buffer('compression_stats', torch.zeros(4))  # [total_compression_ratio, avg_compression_ratio, max_compression_ratio, min_compression_ratio]
        self.register_buffer('sample_count', torch.tensor(0.0))
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, keys: torch.Tensor, values: torch.Tensor, importance: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress using pyramid structure"""
        # Encode keys and values through pyramid
        key_encoded = keys
        value_encoded = values
        
        for encoder in self.encoder_layers:
            key_encoded = F.relu(encoder(key_encoded))
            value_encoded = F.relu(encoder(value_encoded))
        
        # Decode back to original size
        key_decoded = key_encoded
        value_decoded = value_encoded
        
        for decoder in reversed(self.decoder_layers):
            key_decoded = F.relu(decoder(key_decoded))
            value_decoded = F.relu(decoder(value_decoded))
        
        return key_decoded, value_decoded

class SVDCompressor(BaseCompressor):
    """SVD-based compression using singular value decomposition"""
    
    def __init__(self, hidden_size: int, compression_ratio: float = 0.5):
        super().__init__(hidden_size, compression_ratio)
        self.rank = max(int(hidden_size * compression_ratio), 1)
        
        # SVD components
        self.U = None
        self.S = None
        self.V = None
    
    def forward(self, keys: torch.Tensor, values: torch.Tensor, importance: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress using SVD"""
        batch_size, seq_len, hidden_size = keys.shape
        
        # Reshape for SVD
        keys_2d = keys.view(-1, hidden_size)
        values_2d = values.view(-1, hidden_size)
        
        # Compute SVD
        U_k, S_k, V_k = torch.svd(keys_2d)
        U_v, S_v, V_v = torch.svd(values_2d)
        
        # Keep top-k singular values
        U_k = U_k[:, :self.rank]
        S_k = S_k[:self.rank]
        V_k = V_k[:, :self.rank]
        
        U_v = U_v[:, :self.rank]
        S_v = S_v[:self.rank]
        V_v = V_v[:, :self.rank]
        
        # Reconstruct compressed tensors
        keys_compressed = U_k @ torch.diag(S_k) @ V_k.T
        values_compressed = U_v @ torch.diag(S_v) @ V_v.T
        
        # Reshape back
        keys_compressed = keys_compressed.view(batch_size, seq_len, hidden_size)
        values_compressed = values_compressed.view(batch_size, seq_len, hidden_size)
        
        return keys_compressed, values_compressed

class QuantizedCompressor(BaseCompressor):
    """Quantization-based compression"""
    
    def __init__(self, hidden_size: int, compression_ratio: float = 0.5, num_bits: int = 8):
        super().__init__(hidden_size, compression_ratio)
        self.num_bits = num_bits
        self.max_val = 2 ** (num_bits - 1) - 1
        
        # Quantization parameters
        self.register_buffer('key_scale', torch.ones(1))
        self.register_buffer('value_scale', torch.ones(1))
    
    def forward(self, keys: torch.Tensor, values: torch.Tensor, importance: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress using quantization"""
        # Calculate scales
        key_max = torch.max(torch.abs(keys))
        value_max = torch.max(torch.abs(values))
        
        self.key_scale = key_max / self.max_val
        self.value_scale = value_max / self.max_val
        
        # Quantize
        keys_quantized = torch.round(keys / self.key_scale) * self.key_scale
        values_quantized = torch.round(values / self.value_scale) * self.value_scale
        
        return keys_quantized, values_quantized

class FastVCompressor(BaseCompressor):
    """FastV compression using vector quantization"""
    
    def __init__(self, hidden_size: int, compression_ratio: float = 0.5, num_centroids: int = 256, sparsity_threshold: float = 0.1):
        super().__init__(hidden_size, compression_ratio)
        self.num_centroids = num_centroids
        self.sparsity_threshold = sparsity_threshold
        
        # Codebook for vector quantization
        self.codebook = nn.Parameter(torch.randn(num_centroids, hidden_size))
        
        # Sparsity mask
        self.sparsity_mask = None
    
    def forward(self, keys: torch.Tensor, values: torch.Tensor, importance: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress using FastV"""
        batch_size, seq_len, hidden_size = keys.shape
        
        # Reshape for processing
        keys_2d = keys.view(-1, hidden_size)
        values_2d = values.view(-1, hidden_size)
        
        # Find nearest centroids
        distances = torch.cdist(keys_2d, self.codebook)
        key_indices = torch.argmin(distances, dim=1)
        
        distances = torch.cdist(values_2d, self.codebook)
        value_indices = torch.argmin(distances, dim=1)
        
        # Reconstruct using codebook
        keys_compressed = self.codebook[key_indices].view(batch_size, seq_len, hidden_size)
        values_compressed = self.codebook[value_indices].view(batch_size, seq_len, hidden_size)
        
        # Apply sparsity
        if self.sparsity_threshold > 0:
            key_magnitudes = torch.norm(keys_compressed, dim=-1)
            value_magnitudes = torch.norm(values_compressed, dim=-1)
            
            key_mask = key_magnitudes > self.sparsity_threshold
            value_mask = value_magnitudes > self.sparsity_threshold
            
            keys_compressed = keys_compressed * key_mask.unsqueeze(-1)
            values_compressed = values_compressed * value_mask.unsqueeze(-1)
        
        return keys_compressed, values_compressed

class PiKVCompressor(BaseCompressor):
    """
    Unified PiKV compressor with multiple strategies
    Integrates all compression methods with adaptive selection
    """
    def __init__(
        self, 
        hidden_size: int,
        compression_methods: List[str] = ["lora", "pyramid"],
        importance_threshold: float = 0.5,
        adaptive_selection: bool = True
    ):
        super().__init__(hidden_size, compression_ratio=0.5)
        
        self.compression_methods = compression_methods
        self.importance_threshold = importance_threshold
        self.adaptive_selection = adaptive_selection
        
        # Initialize compressors
        self.compressors = nn.ModuleDict()
        self._init_compressors()
        
        # Compression statistics
        self.register_buffer('compression_stats', torch.zeros(len(compression_methods)))
        self.register_buffer('method_usage', torch.zeros(len(compression_methods)))
        self.register_buffer('total_compressions', torch.tensor(0.0))
    
    def _init_compressors(self):
        """Initialize all compression methods"""
        if "lora" in self.compression_methods:
            self.compressors["lora"] = LoRACompressor(self.hidden_size, rank=16)
        
        if "pyramid" in self.compression_methods:
            self.compressors["pyramid"] = PyramidCompressor(self.hidden_size)
        
        if "svd" in self.compression_methods:
            self.compressors["svd"] = SVDCompressor(self.hidden_size)
        
        if "quantized" in self.compression_methods:
            self.compressors["quantized"] = QuantizedCompressor(self.hidden_size)
        
        if "fastv" in self.compression_methods:
            self.compressors["fastv"] = FastVCompressor(self.hidden_size)
    
    def forward(self, keys: torch.Tensor, values: torch.Tensor, importance: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress using selected method"""
        if not self.compressors:
            return keys, values
        
        # Select compression method
        if self.adaptive_selection and importance is not None:
            method_idx = self._select_method_adaptive(importance)
        else:
            method_idx = torch.randint(0, len(self.compression_methods), (1,)).item()
        
        method_name = self.compression_methods[int(method_idx)]
        compressor = self.compressors[method_name]
        
        # Apply compression
        compressed_keys, compressed_values = compressor(keys, values, importance)
        
        # Update statistics
        self._update_stats(int(method_idx), keys, compressed_keys)
        
        return compressed_keys, compressed_values
    
    def _select_method_adaptive(self, importance: torch.Tensor) -> int:
        """Adaptively select compression method based on importance"""
        avg_importance = importance.mean().item()
        
        if avg_importance > self.importance_threshold:
            # High importance - use high-quality compression
            if "lora" in self.compression_methods:
                return self.compression_methods.index("lora")
            elif "pyramid" in self.compression_methods:
                return self.compression_methods.index("pyramid")
        else:
            # Low importance - use high-compression methods
            if "fastv" in self.compression_methods:
                return self.compression_methods.index("fastv")
            elif "quantized" in self.compression_methods:
                return self.compression_methods.index("quantized")
        
        # Default to first method
        return 0
    
    def _update_stats(self, method_idx: int, original: torch.Tensor, compressed: torch.Tensor):
        """Update compression statistics"""
        compression_ratio = compressed.numel() / original.numel()
        self.compression_stats[method_idx] = compression_ratio
        self.method_usage[method_idx] += 1
        self.total_compressions += 1
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics"""
        stats = {
            "total_compressions": self.total_compressions.item(),
            "usage": {}
        }
        
        # Method usage statistics
        for i, method in enumerate(self.compression_methods):
            if self.total_compressions > 0:
                usage_ratio = self.method_usage[i].item() / self.total_compressions.item()
            else:
                usage_ratio = 0.0
            
            stats["usage"][method] = usage_ratio
            stats[method] = {
                "compression_ratio": self.compression_stats[i].item(),
                "usage_count": self.method_usage[i].item()
            }
        
        return stats

# Factory function for creating compressors
def create_compressor(
    compressor_type: str,
    hidden_size: int,
    **kwargs
) -> BaseCompressor:
    """Create compressor instance based on type"""
    
    compressor_map = {
        'lora': LoRACompressor,
        'pyramid': PyramidCompressor,
        'svd': SVDCompressor,
        'quantized': QuantizedCompressor,
        'fastv': FastVCompressor,
        'pikv': PiKVCompressor
    }
    
    if compressor_type not in compressor_map:
        raise ValueError(f"Unsupported compressor type: {compressor_type}. "
                        f"Supported types: {list(compressor_map.keys())}")
    
    return compressor_map[compressor_type](hidden_size, **kwargs) 