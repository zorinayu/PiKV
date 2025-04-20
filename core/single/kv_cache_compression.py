import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PyramidCompression(nn.Module):
    def __init__(self, hidden_size, compression_ratio=0.5):
        super(PyramidCompression, self).__init__()
        self.hidden_size = hidden_size
        self.compression_ratio = compression_ratio
        
        # Initialize compression layers
        self.compression_layers = nn.ModuleList([
            nn.Linear(hidden_size, int(hidden_size * compression_ratio)),
            nn.Linear(int(hidden_size * compression_ratio), hidden_size)
        ])
        
        # Initialize importance weights
        self.importance_weights = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, x, importance=None):
        """
        Apply pyramid compression to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            importance: Optional importance scores of shape [batch_size, seq_len]
        
        Returns:
            Compressed tensor of same shape as input
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Apply importance weighting if provided
        if importance is not None:
            x = x * importance.unsqueeze(-1)
        
        # Apply compression
        compressed = self.compression_layers[0](x)
        decompressed = self.compression_layers[1](compressed)
        
        # Residual connection
        output = x + decompressed
        
        return output

class DynamicCompression(nn.Module):
    def __init__(self, hidden_size, min_ratio=0.1, max_ratio=1.0):
        super(DynamicCompression, self).__init__()
        self.hidden_size = hidden_size
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        
        # Initialize compression layers
        self.compression_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        ])
        
        # Initialize compression ratio predictor
        self.ratio_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, importance=None):
        """
        Apply dynamic compression based on input characteristics.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            importance: Optional importance scores of shape [batch_size, seq_len]
        
        Returns:
            Compressed tensor of same shape as input
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Predict compression ratio
        if importance is not None:
            # Use importance scores to predict ratio
            ratio = self.ratio_predictor(importance.unsqueeze(-1))
        else:
            # Use input features to predict ratio
            ratio = self.ratio_predictor(x.mean(dim=1))
        
        # Scale ratio to desired range
        ratio = self.min_ratio + (self.max_ratio - self.min_ratio) * ratio
        
        # Apply compression
        compressed = self.compression_layers[0](x)
        decompressed = self.compression_layers[1](compressed)
        
        # Mix original and compressed features based on ratio
        output = ratio * x + (1 - ratio) * decompressed
        
        return output

class KVCacheCompressor(nn.Module):
    def __init__(self, hidden_size, compression_type='pyramid', **kwargs):
        super(KVCacheCompressor, self).__init__()
        self.hidden_size = hidden_size
        
        if compression_type == 'pyramid':
            self.compressor = PyramidCompression(hidden_size, **kwargs)
        elif compression_type == 'dynamic':
            self.compressor = DynamicCompression(hidden_size, **kwargs)
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
        
    def forward(self, keys, values, importance=None):
        """
        Compress key-value cache.
        
        Args:
            keys: Key tensor of shape [batch_size, seq_len, hidden_size]
            values: Value tensor of shape [batch_size, seq_len, hidden_size]
            importance: Optional importance scores of shape [batch_size, seq_len]
        
        Returns:
            Compressed keys and values
        """
        compressed_keys = self.compressor(keys, importance)
        compressed_values = self.compressor(values, importance)
        
        return compressed_keys, compressed_values 