import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

distributed_config = {
    'world_size': 8,  # 总GPU数量改为8
    'rank': 0,  # 当前进程的rank
    'dist_backend': 'nccl',  # 分布式后端
    'dist_url': 'tcp://localhost:23456',  # 分布式通信地址
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'local_rank': 0,  # 本地GPU编号
    'expert_parallel': True,  # 是否使用专家并行
    'tensor_parallel': True,  # 是否使用张量并行
    'pipeline_parallel': True,  # 是否使用流水线并行
    'num_micro_batches': 8,  # 流水线并行的微批次数量改为8
    'expert_placement': 'balanced',  # 专家放置策略：'balanced' 或 'greedy'
    'communication_overlap': True,  # 是否启用通信重叠
    'gradient_accumulation_steps': 1,  # 梯度累积步数
    'checkpoint_interval': 100,  # 检查点保存间隔
    'use_mixed_precision': True,  # 是否使用混合精度训练
} 