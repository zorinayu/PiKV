import torch
import torch.nn as nn
from .config import config

class ExternalMemoryCache(nn.Module):
    """
    External memory cache using CXL-based memory disaggregation.
    """
    def __init__(self):
        super(ExternalMemoryCache, self).__init__()
        self.cache = {}
        self.max_size = config.get('external_cache_size', 1000000)
    
    def get(self, key):
        return self.cache.get(key)
    
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear() 