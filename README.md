# PiKV: Efficient KV Cache Management for Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

PiKV is a high-performance KV cache management system designed for Large Language Models (LLMs). It provides efficient compression, scheduling, and routing strategies to optimize memory usage and inference performance.

## Features

- **Unified MoE Implementation**: Consolidated mixture-of-experts with multiple routing strategies
- **Advanced Compression**: Multiple compression methods (LoRA, Pyramid, SVD, Quantization)
- **Smart Caching**: Intelligent cache scheduling with multiple policies
- **Multimodal Support**: Flex-MoE for handling arbitrary modality combinations
- **Time Series Optimization**: Time-MoE for temporal data processing
- **Distributed Training**: Support for large-scale distributed training

## Quick Start

### Installation

```bash
git clone https://github.com/your-username/PiKV.git
cd PiKV
pip install -e .
```

### Basic Usage

```python
from core.single.moe import create_moe

# Create a basic MoE
moe = create_moe('base', hidden_size=1024, num_experts=8, top_k=2)

# Create a Flex-MoE for multimodal learning
flex_moe = create_moe('flex', hidden_size=1024, num_experts=16, top_k=4)

# Create a Time-MoE for time series
time_moe = create_moe('time', hidden_size=1024, num_experts=8, top_k=2)

# Create a PiKV MoE with LoRA and distillation
pikv_moe = create_moe('pikv', hidden_size=1024, num_experts=8, top_k=2, 
                      rank=16, alpha=1.0, use_distillation=True)
```

### Examples

Run the unified MoE example:

```bash
python examples/moe_example.py
```

## Architecture

### Core Components

- **MoE Module** (`core/single/moe.py`): Unified mixture-of-experts implementation
- **Compression Module** (`core/single/module/pikv_compression.py`): Various compression strategies
- **Scheduling Module** (`core/single/module/pikv_scheduling.py`): Cache scheduling policies
- **Routing Module** (`core/single/module/pikv_routing.py`): Advanced routing strategies

### MoE Strategies

1. **Base MoE**: Standard mixture-of-experts implementation
2. **Flex-MoE**: Multimodal learning with flexible routing
3. **Time-MoE**: Time series prediction with temporal awareness
4. **PiKV MoE**: Enhanced with LoRA and knowledge distillation

### Compression Methods

- **LoRA**: Low-rank adaptation for efficient fine-tuning
- **Pyramid**: Multi-level compression with importance weighting
- **SVD**: Singular value decomposition compression
- **Quantization**: Reduced precision for memory efficiency

### Cache Policies

- **LRU**: Least recently used eviction
- **H2O**: Heavy-hitter and recent object optimization
- **QUEST**: Quality-aware eviction strategy

## Performance

PiKV provides significant improvements in:

- **Memory Usage**: Up to 80% reduction in KV cache memory
- **Inference Speed**: 2-5x faster inference with optimized caching
- **Training Efficiency**: Efficient distributed training with load balancing
- **Model Quality**: Maintained performance with advanced compression

## Citation

If you use PiKV in your research, please cite:

```bibtex
@article{liu2025pikv,
  title={PiKV: Efficient KV Cache Management for Large Language Models},
  author={Liu, Dong and others},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Flex-MoE: [UNITES-Lab/Flex-MoE](https://github.com/UNITES-Lab/Flex-MoE)
- Time-MoE: [Time-MoE/Time-MoE](https://github.com/Time-MoE/Time-MoE)
- FastMoE: [laekov/fastmoe](https://github.com/laekov/fastmoe)
- Mixture of Experts: [lucidrains/mixture-of-experts](https://github.com/lucidrains/mixture-of-experts)

