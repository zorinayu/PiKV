# PiKV Project Structure

## Overview
PiKV has been reorganized for better maintainability and clarity. This document outlines the new structure.

## Directory Structure

```
PiKV/
├── README.md                 # Main project documentation
├── LICENSE                   # MIT License
├── setup.py                 # Package installation
├── requirements.txt          # Dependencies
├── PROJECT_STRUCTURE.md      # This file
│
├── core/                     # Core implementation
│   ├── single/              # Single-node implementation
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration parameters
│   │   ├── moe.py           # Unified MoE implementation
│   │   ├── distillation.py  # Knowledge distillation
│   │   ├── utils.py         # Utility functions
│   │   ├── main.py          # Main entry point
│   │   │
│   │   ├── module/          # Core modules
│   │   │   ├── __init__.py
│   │   │   ├── pikv_compression.py    # Compression strategies
│   │   │   ├── pikv_scheduling.py     # Cache scheduling
│   │   │   └── pikv_routing.py       # Advanced routing
│   │   │
│   │   └── model_compression/        # Model compression
│   │       └── README.md
│   │
│   ├── distributed/         # Distributed implementation
│   └── cuda/               # CUDA kernels
│
├── examples/                # Usage examples
│   └── moe_example.py      # Unified MoE examples
│
├── downstream_tasks/        # Downstream applications
│   ├── llm/                # Language model tasks
│   └── vision/             # Computer vision tasks
│
├── data/                    # Data files
├── assets/                  # Images and resources
└── output/                  # Generated outputs
```

## Key Changes Made

### 1. **Consolidated MoE Implementation**
- **Before**: Multiple separate MoE files (`pikv_moe.py`, `routing_moe.py`, `lora_moe.py`)
- **After**: Single unified `moe.py` with all routing strategies

### 2. **Simplified Examples**
- **Before**: 5+ separate example files with overlapping functionality
- **After**: Single `moe_example.py` demonstrating all MoE types

### 3. **Cleaned Configuration**
- **Before**: Complex config with unused parameters
- **After**: Essential parameters only, clearly organized

### 4. **Removed Duplicates**
- Deleted test files (moved to proper test directory)
- Removed duplicate README files
- Consolidated similar functionality

## Core Components

### **MoE Module** (`core/single/moe.py`)
- `BaseRouter`: Standard routing logic
- `FlexMoERouter`: Multimodal learning support
- `TimeMoERouter`: Time series optimization
- `MoE`: Unified expert mixture implementation
- `PiKVMoE`: Enhanced with LoRA and distillation

### **Compression Module** (`core/single/module/pikv_compression.py`)
- Multiple compression strategies
- Adaptive compression selection
- Performance monitoring

### **Scheduling Module** (`core/single/module/pikv_scheduling.py`)
- Multiple cache policies (LRU, H2O, QUEST)
- Dynamic policy switching
- Load balancing

### **Routing Module** (`core/single/module/pikv_routing.py`)
- Advanced routing strategies
- Load balancing algorithms
- Hierarchical routing

## Usage

### **Basic MoE**
```python
from core.single.moe import create_moe

moe = create_moe('base', hidden_size=1024, num_experts=8)
```

### **Flex-MoE for Multimodal**
```python
flex_moe = create_moe('flex', hidden_size=1024, num_experts=16)
```

### **Time-MoE for Time Series**
```python
time_moe = create_moe('time', hidden_size=1024, num_experts=8)
```

### **PiKV MoE with Extensions**
```python
pikv_moe = create_moe('pikv', hidden_size=1024, num_experts=8, 
                      rank=16, alpha=1.0, use_distillation=True)
```

## Benefits of Reorganization

1. **Maintainability**: Single source of truth for each component
2. **Clarity**: Clear separation of concerns
3. **Efficiency**: No duplicate code or functionality
4. **Usability**: Simple, consistent API across all MoE types
5. **Documentation**: Single, comprehensive documentation

## Migration Guide

If you were using the old structure:

1. **MoE Usage**: Replace imports from `pikv_moe.py` with `moe.py`
2. **Examples**: Use the new unified `moe_example.py`
3. **Configuration**: Update to use the simplified `config.py`
4. **Routing**: All routing strategies are now in the unified MoE

The new structure maintains backward compatibility while providing a cleaner, more maintainable codebase.
