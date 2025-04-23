import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class KVCacheCompressor(nn.Module):
    def __init__(self, hidden_size: int, compression_ratio: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.compression_ratio = compression_ratio
        self.compressed_size = int(hidden_size * compression_ratio)
        
        # Initialize compression layers
        self.compressor = nn.Linear(hidden_size, self.compressed_size)
        self.decompressor = nn.Linear(self.compressed_size, hidden_size)
        
        # Initialize importance projection
        self.importance_proj = nn.Linear(hidden_size, 1)
    
    def compress(self, x: torch.Tensor, importance: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project to compressed space
        compressed = self.compressor(x)
        
        # Calculate importance if not provided
        if importance is None:
            importance = torch.sigmoid(self.importance_proj(x))
        
        return compressed, importance
    
    def decompress(self, compressed: torch.Tensor, importance: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Decompress back to original space
        decompressed = self.decompressor(compressed)
        
        # Apply importance if provided
        if importance is not None:
            decompressed = decompressed * importance
        
        return decompressed
    
    def forward(self, x: torch.Tensor, importance: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compress input
        compressed, importance = self.compress(x, importance)
        
        # Decompress back
        decompressed = self.decompress(compressed, importance)
        
        return decompressed, importance 