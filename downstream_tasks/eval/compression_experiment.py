import torch
from core.single.module.pikv_compression import PyramidCompressor, LoRACompressor, SVDCompressor

# Initialize compressors
pyramid_compressor = PyramidCompressor(hidden_size=512)
lora_compressor = LoRACompressor(hidden_size=512)
svd_compressor = SVDCompressor(hidden_size=512)

# Dummy data
keys = torch.randn(10, 512)
values = torch.randn(10, 512)

# Evaluate compression
compressed_keys, compressed_values = pyramid_compressor(keys, values)
print("Pyramid Compression:", compressed_keys.shape, compressed_values.shape)

compressed_keys, compressed_values = lora_compressor(keys, values)
print("LoRA Compression:", compressed_keys.shape, compressed_values.shape)

compressed_keys, compressed_values = svd_compressor(keys, values)
print("SVD Compression:", compressed_keys.shape, compressed_values.shape) 