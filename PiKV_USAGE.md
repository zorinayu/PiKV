# PiKV使用指南

## 目录
1. [简介](#简介)
2. [核心模块](#核心模块)
   - [PiKV MoE](#pikv-moe)
   - [PiKV Compression](#pikv-compression)
   - [PiKV Scheduling](#pikv-scheduling)
3. [与预训练模型集成](#与预训练模型集成)
4. [最佳实践](#最佳实践)

## 简介

PiKV是一个高效的KV缓存管理系统，专为大型语言模型(LLM)设计。它提供了多种压缩和优化策略，可以显著减少内存使用并提高推理性能。本文档将详细介绍PiKV的三大核心模块的使用方法。

## 核心模块

### PiKV MoE

PiKV MoE是核心的路由和专家混合模块，负责高效地管理和分配计算资源。

```python
from core.single.pikv_moe import PiKVMoE
from core.single.config import config

# 基础用法
pikv_moe = PiKVMoE(
    rank=4,                # LoRA秩
    alpha=1.0,            # LoRA alpha
    use_distillation=True, # 启用知识蒸馏
    teacher_hidden_size=config['hidden_size'] * 2
)

# 分布式用法
from core.single.pikv_moe import DistributedPiKVMoE
distributed_pikv = DistributedPiKVMoE(
    rank=4,
    alpha=1.0,
    use_distillation=True
)

# 使用示例
def process_with_pikv(input_tensor):
    # 处理输入
    output = pikv_moe(input_tensor)
    return output
```

### MoE策略集成

PiKV支持集成多种先进的MoE策略，以满足不同应用场景的需求。这些策略实现在 `core/single/moe_strategies_fixed.py` 文件中。

```python
from core.single.moe_strategies_fixed import create_moe_router

# 创建不同类型的MoE路由器
flex_router = create_moe_router('flex', hidden_size=1024, num_experts=16, top_k=4)
time_router = create_moe_router('time', hidden_size=1024, num_experts=8, top_k=2)
fast_router = create_moe_router('fast', hidden_size=1024, num_experts=16, top_k=2)
mixture_router = create_moe_router('mixture', hidden_size=1024, num_experts=8, top_k=2)

# 使用示例
def process_with_moe_strategies(input_tensor, strategy_type='flex'):
    router = create_moe_router(strategy_type, hidden_size=1024, num_experts=8)
    
    if strategy_type == 'flex':
        # 多模态数据
        modality_info = {
            'image': torch.randn(4, 128, 1024),
            'genomic': torch.randn(4, 128, 1024)
        }
        dispatch, combine, probs, loss = router(input_tensor, modality_info)
    elif strategy_type == 'time':
        # 时间序列数据
        time_info = {
            'timestamps': torch.arange(128).float(),
            'seasonality': torch.sin(torch.arange(128) * 2 * torch.pi / 24)
        }
        dispatch, combine, probs, loss = router(input_tensor, time_info)
    else:
        # 标准路由
        dispatch, combine, probs, loss = router(input_tensor)
    
    return dispatch, combine, probs, loss
```

### PiKV Compression

PiKV Compression提供了多种压缩策略，用于优化KV缓存的内存使用。

```python
from core.single.module.pikv_compression import (
    PiKVCompressor,
    LoRACompressor,
    PyramidCompressor,
    FastVCompressor,
    SVDCompressor,
    QuantizedCompressor
)

# 综合压缩器
compressor = PiKVCompressor(
    hidden_size=1024,
    compression_methods=["lora", "pyramid", "fastv"],
    importance_threshold=0.5,
    adaptive_selection=True
)

# LoRA压缩
lora_compressor = LoRACompressor(
    hidden_size=1024,
    rank=64,
    alpha=1.0,
    dropout=0.1
)

# 金字塔压缩
pyramid_compressor = PyramidCompressor(
    hidden_size=1024,
    compression_ratio=0.5,
    num_levels=3,
    decay_factor=0.8
)

# 使用示例
def compress_kv_cache(keys, values, importance=None):
    # 使用综合压缩器
    compressed_keys, compressed_values = compressor(keys, values, importance)
    
    # 获取压缩统计信息
    stats = compressor.get_compression_stats()
    print(f"压缩率: {stats['compression_ratio']:.2f}")
    
    return compressed_keys, compressed_values
```

### PiKV Scheduling

PiKV Scheduling提供了多种缓存调度策略，用于优化内存使用和访问效率。

```python
from core.single.module.pikv_scheduling import (
    SchedulingPolicy,
    CacheSchedulingManager,
    H2OScheduler,
    LRUScheduler,
    QUESTScheduler
)

# 缓存调度管理器
cache_manager = CacheSchedulingManager(
    cache_size=4096,
    hidden_size=1024,
    policy=SchedulingPolicy.LRU
)

# H2O调度器
h2o_scheduler = H2OScheduler(
    cache_size=4096,
    hidden_size=1024,
    heavy_ratio=0.1,
    recent_ratio=0.1
)

# 使用示例
def manage_cache(keys, values, metadata=None):
    # 更新缓存
    cache_manager.update_cache(keys, values, metadata)
    
    # 获取缓存统计信息
    stats = cache_manager.get_cache_stats()
    print(f"命中率: {stats['hit_rate']:.2f}")
    
    # 获取缓存数据
    cached_keys, cached_values = cache_manager.get_cached_data()
    return cached_keys, cached_values
```

## 与预训练模型集成

以下是如何将PiKV与预训练模型（如Qwen）集成的完整示例：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.single.pikv_moe import PiKVMoE
from core.single.module.pikv_compression import PiKVCompressor
from core.single.module.pikv_scheduling import CacheSchedulingManager, SchedulingPolicy
from core.single.config import config

class PiKVEnhancedModel:
    def __init__(self, model_name="Qwen/Qwen-7B", max_length=1024):
        # 初始化模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # 获取模型配置
        self.hidden_size = self.model.config.hidden_size
        config['hidden_size'] = self.hidden_size
        
        # 初始化PiKV组件
        self.pikv_moe = PiKVMoE(
            rank=4,
            alpha=1.0,
            use_distillation=True,
            teacher_hidden_size=self.hidden_size * 2
        )
        
        self.compressor = PiKVCompressor(
            hidden_size=self.hidden_size,
            compression_methods=["lora", "pyramid"],
            importance_threshold=0.5
        )
        
        self.cache_manager = CacheSchedulingManager(
            cache_size=4096,
            hidden_size=self.hidden_size,
            policy=SchedulingPolicy.LRU
        )
        
        # 初始化缓存
        self.kv_cache = {}
        self.current_length = 0
    
    def generate(self, prompt: str, max_new_tokens: int = 50):
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        input_ids = input_ids.to(self.model.device)
        
        # 初始化输出
        output_ids = input_ids.clone()
        self.current_length = input_ids.size(1)
        
        # 生成token
        for _ in range(max_new_tokens):
            # 获取模型输出
            outputs = self.model(
                input_ids=output_ids,
                use_cache=True,
                return_dict=True
            )
            
            # 处理每一层的KV缓存
            new_past_key_values = []
            for layer_idx, layer_output in enumerate(outputs.past_key_values):
                key, value = layer_output
                
                # 1. 通过PiKV MoE处理
                processed_key = self.pikv_moe(key)
                processed_value = self.pikv_moe(value)
                
                # 2. 压缩KV缓存
                compressed_key, compressed_value = self.compressor(
                    processed_key,
                    processed_value
                )
                
                # 3. 更新缓存调度
                self.cache_manager.update_cache(
                    compressed_key,
                    compressed_value,
                    metadata={'layer_idx': layer_idx}
                )
                
                # 获取处理后的KV缓存
                cached_key, cached_value = self.cache_manager.get_cached_data()
                new_past_key_values.append((cached_key, cached_value))
            
            # 更新模型的KV缓存
            outputs.past_key_values = tuple(new_past_key_values)
            
            # 获取下一个token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # 更新输出
            output_ids = torch.cat([output_ids, next_token.unsqueeze(-1)], dim=-1)
            
            # 检查是否生成了结束token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # 解码输出
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text
    
    def get_stats(self):
        """获取各种统计信息"""
        return {
            'compression_stats': self.compressor.get_compression_stats(),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'moe_stats': self.pikv_moe.get_stats()
        }

# 使用示例
def main():
    # 初始化增强模型
    model = PiKVEnhancedModel(model_name="Qwen/Qwen-7B")
    
    # 生成文本
    prompt = "请介绍一下人工智能的发展历史"
    generated_text = model.generate(prompt, max_new_tokens=100)
    print(f"生成的文本: {generated_text}")
    
    # 获取统计信息
    stats = model.get_stats()
    print("\n统计信息:")
    print(f"压缩率: {stats['compression_stats']['compression_ratio']:.2f}")
    print(f"缓存命中率: {stats['cache_stats']['hit_rate']:.2f}")
```

## 最佳实践

1. **选择合适的压缩策略**
   - 对于较小的模型，可以使用LoRA压缩
   - 对于较大的模型，建议使用金字塔压缩或FastV压缩
   - 可以根据具体任务动态选择压缩策略

2. **优化缓存调度**
   - 使用LRU策略处理一般场景
   - 使用H2O策略处理长序列
   - 使用DuoAttention, QUEST策略处理需要高质量输出的场景

3. **内存管理**
   - 定期监控内存使用情况
   - 根据实际需求调整缓存大小
   - 使用压缩统计信息优化压缩参数

4. **性能优化**
   - 在推理时关闭蒸馏
   - 使用批处理提高吞吐量
   - 根据硬件条件调整并行度

5. **监控和调试**
   - 定期检查压缩率
   - 监控缓存命中率
   - 分析性能瓶颈

6. **分布式部署**
   - 使用分布式PiKV MoE进行大规模部署
   - 合理分配计算资源
   - 注意同步和通信开销 