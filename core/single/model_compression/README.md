# PiKV Model Compression Framework

This module provides a comprehensive set of compression techniques for KV cache in large language models. The framework includes three main categories of compression methods, each with multiple implementations designed to reduce memory usage while maintaining model performance.

## Overview

The PiKV Model Compression framework offers:

1. **Matrix Defactorization**: Techniques that use low-rank approximation to reduce parameter count
2. **Cache Reduction**: Methods to directly compress KV cache through various strategies
3. **Distillation**: Approaches that use knowledge distillation to compress cache representations

All compressors share a common interface, making them easy to integrate into existing workflows. Each compressor also provides detailed statistics to help evaluate and compare different compression methods.

## Installation

The module is included as part of the PiKV package. No additional installation is required.

## Usage

### Basic Usage

Here's a simple example of using a compressor:

```python
import torch
from core.single.model_compression import PyramidCompressor

# Initialize compressor
compressor = PyramidCompressor(
    hidden_size=256,  # Match your model's hidden size
    compression_ratio=0.5,
    num_levels=3
)

# Apply compression to KV cache
keys = torch.randn(1, 128, 256)  # [batch_size, seq_len, hidden_size]
values = torch.randn(1, 128, 256)
importance = torch.rand(1, 128)  # Optional importance scores

# Compress KV cache
compressed_keys, compressed_values = compressor(keys, values, importance)

# Get compression statistics
stats = compressor.get_compression_stats()
print(f"Compression ratio: {stats['compression_ratio']:.2f}")
print(f"Memory reduction: {stats['memory_reduction']:.2%}")
```

### Evaluating Compressors

The module includes evaluation utilities to benchmark and compare different compression methods:

```python
from core.single.model_compression import CompressionEvaluator
from core.single.model_compression import LoRACompressor, PyramidCompressor

# Initialize evaluator
evaluator = CompressionEvaluator(hidden_size=256)

# Initialize compressors
compressors = {
    "LoRA": LoRACompressor(hidden_size=256, rank=16),
    "Pyramid": PyramidCompressor(hidden_size=256, compression_ratio=0.5)
}

# Compare compressors
results = evaluator.compare_compressors(compressors)

# Save results
evaluator.save_results("compression_results.csv")
```

### Next Token Prediction Test

You can evaluate how compression affects model performance on next token prediction:

```python
from core.single.model_compression import SimpleNextTokenPredictor

# Initialize predictor
predictor = SimpleNextTokenPredictor(hidden_size=256, vocab_size=50257)

# Run next token prediction test
prediction_results = evaluator.run_next_token_prediction_test(
    compressors=compressors,
    predictor=predictor
)
```

## Implemented Compressors

### Matrix Defactorization

1. **LoRACompressor**:
   - Implements Low-Rank Adaptation for KV cache compression
   - Configurable rank and scaling parameter
   - Maintains high fidelity through residual connections

2. **LoRAPlusCompressor**:
   - Enhanced version of LoRA with adaptive rank selection
   - Adjusts compression based on token importance
   - Supports multiple ranks for different importance levels

### Cache Reduction

1. **PyramidCompressor**:
   - Implements a pyramid structure for hierarchical compression
   - Applies different compression levels based on token importance
   - Distributes tokens across multiple compression levels

2. **FastVCompressor**:
   - Centroid-based compression for similar values
   - Dynamic cluster updates based on usage patterns
   - Sparse residual updates for high-importance tokens

### Distillation

1. **FastVideoCompressor**:
   - Video compression inspired approach for sequential tokens
   - Keyframe selection with motion prediction for intermediate tokens
   - Temporal correlation modeling

2. **MiniLLMCompressor**:
   - Student-teacher model for KV cache compression
   - Small student model compresses teacher representations
   - Optional attention mechanism for improved fidelity

## Extending the Framework

You can create your own compressor by extending the `BaseCompressor` class:

```python
from core.single.model_compression.matrix_defactorization import BaseCompressor

class MyCustomCompressor(BaseCompressor):
    def __init__(self, hidden_size, my_param=0.5):
        super(MyCustomCompressor, self).__init__(hidden_size)
        self.my_param = my_param
        # Initialize your compressor
        
    def forward(self, keys, values, importance=None):
        # Implement your compression algorithm
        # ...
        return compressed_keys, compressed_values
        
    def get_compression_stats(self):
        stats = super().get_compression_stats()
        # Add your custom statistics
        stats.update({
            "my_param": self.my_param
        })
        return stats
```

## Evaluation and Benchmarking

The framework includes comprehensive evaluation tools:

1. **Performance Metrics**:
   - Compression ratio
   - Memory reduction
   - Compression time
   - MSE loss
   - Cosine similarity

2. **Next Token Prediction**:
   - Accuracy impact
   - Compression vs. accuracy tradeoff

3. **Visualization**:
   - Performance comparison plots
   - Tradeoff analysis

## Running Tests

To run a comprehensive test of all compressors:

```bash
python core/single/model_compression/test_compression.py
```

This will evaluate all compressors on various metrics and save the results to the `results/` directory.