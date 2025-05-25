# PiKV 分布式知识蒸馏使用指南

## 概述

本文档介绍如何使用 PiKV 的分布式知识蒸馏功能进行 next token prediction 训练。知识蒸馏允许较小的学生模型从较大的教师模型中学习，提高性能的同时保持效率。

## 快速开始

### 1. 基本的分布式蒸馏训练

```bash
torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=23459 \
    downstream_tasks/llm/next_tok_pred/d_transformers_distillation.py \
    --use_distillation \
    --model gpt2 \
    --max_tokens 100
```

### 2. 自定义教师模型配置

```bash
torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=23459 \
    downstream_tasks/llm/next_tok_pred/d_transformers_distillation.py \
    --use_distillation \
    --model gpt2 \
    --teacher_hidden_size 1536 \
    --distillation_temperature 5.0 \
    --distillation_alpha 0.8 \
    --max_tokens 100
```

### 3. 保存和加载检查点

```bash
# 训练并保存检查点
torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=23459 \
    downstream_tasks/llm/next_tok_pred/d_transformers_distillation.py \
    --use_distillation \
    --model gpt2 \
    --save_checkpoint ./checkpoints/pikv_distill_model.pth

# 从检查点继续训练
torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=23459 \
    downstream_tasks/llm/next_tok_pred/d_transformers_distillation.py \
    --use_distillation \
    --load_checkpoint ./checkpoints/pikv_distill_model.pth
```

## 参数说明

### 必需参数

- `--use_distillation`: 启用知识蒸馏功能

### 可选参数

- `--model`: 基础模型名称或路径 (默认: "gpt2")
- `--teacher_hidden_size`: 教师模型的隐藏层大小 (默认: 学生模型的2倍)
- `--distillation_temperature`: 蒸馏温度，控制软标签的平滑程度 (默认: 4.0)
- `--distillation_alpha`: 蒸馏损失的权重 (默认: 0.7)
- `--max_tokens`: 生成的最大token数量 (默认: 50)
- `--save_checkpoint`: 保存检查点的路径
- `--load_checkpoint`: 加载检查点的路径

### 分布式参数

- `--nproc_per_node`: 每个节点的进程数 (通常等于GPU数量)
- `--nnodes`: 节点数量
- `--node_rank`: 当前节点的排名
- `--master_addr`: 主节点地址
- `--master_port`: 主节点端口

## 功能特性

### 1. 知识蒸馏

- **多层次蒸馏**: 支持logits、特征、专家输出和KV缓存的蒸馏
- **自适应温度**: 可调节的蒸馏温度参数
- **动态权重**: 可配置的损失权重平衡

### 2. 分布式训练

- **多GPU支持**: 自动分布式环境初始化
- **容错机制**: 完善的错误处理和恢复
- **同步训练**: 确保所有进程同步

### 3. PiKV集成

- **压缩缓存**: 集成PiKV的KV缓存压缩技术
- **LoRA适配**: 支持低秩适配的高效微调
- **专家路由**: 智能的专家选择和负载均衡

## 使用示例

### 示例1: 基础蒸馏训练

```python
# 在Python代码中使用
from downstream_tasks.llm.next_tok_pred.d_transformers_distillation import DistributedPiKVCacheWithDistillation

# 初始化
pikv_cache = DistributedPiKVCacheWithDistillation(
    model_name="gpt2",
    use_distillation=True,
    teacher_hidden_size=1536,
    distillation_temperature=4.0
)

# 生成文本
text = pikv_cache.generate_with_distillation(
    "The future of AI is",
    max_new_tokens=50,
    use_teacher=True
)
print(text)
```

### 示例2: 训练步骤

```python
import torch

# 创建优化器
optimizer = torch.optim.Adam(pikv_cache.model.parameters(), lr=1e-4)

# 准备训练数据
input_data = torch.randint(0, 1000, (4, 20), device=pikv_cache.device)
targets = torch.randint(0, 1000, (4, 20), device=pikv_cache.device)

# 执行蒸馏训练步骤
loss_info = pikv_cache.distillation_training_step(
    input_data=input_data,
    targets=targets,
    optimizer=optimizer
)

print("训练损失:", loss_info)
```

## 性能优化建议

### 1. 硬件配置

- **GPU内存**: 建议至少8GB显存，支持教师和学生模型同时加载
- **CPU**: 多核CPU有助于数据预处理和分布式通信
- **网络**: 高带宽网络连接对多节点训练很重要

### 2. 参数调优

- **蒸馏温度**: 
  - 较高温度(4-6)产生更平滑的概率分布
  - 较低温度(2-3)保留更多原始信息
- **学习率**: 建议从1e-4开始，根据收敛情况调整
- **批次大小**: 根据GPU内存调整，建议每GPU 2-4个样本

### 3. 监控指标

- **蒸馏损失**: 监控学生模型学习教师知识的效果
- **生成质量**: 定期评估生成文本的质量
- **内存使用**: 监控GPU内存使用情况

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小或教师模型大小
   --teacher_hidden_size 1024
   ```

2. **分布式初始化失败**
   ```bash
   # 检查端口是否被占用，尝试不同端口
   --master_port 23460
   ```

3. **模型加载错误**
   ```bash
   # 确保模型路径正确，或使用默认模型
   --model gpt2
   ```

### 调试模式

```bash
# 启用详细日志
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 单GPU测试
torchrun --nproc_per_node=1 \
    downstream_tasks/llm/next_tok_pred/d_transformers_distillation.py \
    --use_distillation \
    --model gpt2
```

## 高级用法

### 1. 多节点训练

```bash
# 节点0 (主节点)
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=23459 \
    downstream_tasks/llm/next_tok_pred/d_transformers_distillation.py \
    --use_distillation

# 节点1
torchrun --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=23459 \
    downstream_tasks/llm/next_tok_pred/d_transformers_distillation.py \
    --use_distillation
```

### 2. 自定义蒸馏策略

可以通过修改 `DistributedPiKVCacheWithDistillation` 类来实现自定义的蒸馏策略：

```python
class CustomDistillation(DistributedPiKVCacheWithDistillation):
    def _get_teacher_outputs(self, input_ids):
        # 自定义教师模型输出逻辑
        pass
    
    def distillation_training_step(self, input_data, targets=None, optimizer=None):
        # 自定义训练步骤
        pass
```

## 参考资料

- [Knowledge Distillation原理](https://arxiv.org/abs/1503.02531)
- [PyTorch分布式训练指南](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PiKV技术文档](../../core/single/DISTILLATION_README.md)