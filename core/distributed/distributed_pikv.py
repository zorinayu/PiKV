import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from distributed_config import distributed_config as dconfig
from config import config

class DistributedExpert(nn.Module):
    def __init__(self, expert_id, world_size):
        super(DistributedExpert, self).__init__()
        self.expert_id = expert_id
        self.world_size = world_size
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        
    def forward(self, x):
        return F.relu(self.dense(x))

class DistributedPiKVMoE(nn.Module):
    def __init__(self):
        super(DistributedPiKVMoE, self).__init__()
        self.world_size = dconfig['world_size']
        self.rank = dconfig['rank']
        
        # 专家并行：每个GPU只负责一部分专家
        experts_per_gpu = config['num_experts'] // self.world_size
        self.local_experts = nn.ModuleList([
            DistributedExpert(i + self.rank * experts_per_gpu, self.world_size)
            for i in range(experts_per_gpu)
        ])
        
        # 路由层在所有GPU上复制
        self.gate = nn.Linear(config['hidden_size'], config['num_experts'])
        
        # 缓存大小分配
        self.cache_sizes = self.pyramidal_cache_allocation()
        
        # 混合精度训练 - 移到这里，因为这是模型的一部分
        self.use_mixed_precision = dconfig['use_mixed_precision']
        
    def pyramidal_cache_allocation(self):
        C1 = config['kv_cache_size']
        d = config['cache_decrement']
        return [C1 - (i - 1) * d for i in range(1, config['num_layers'] + 1)]
    
    def forward(self, x):
        # 计算路由分数
        gate_scores = self.gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        # 专家并行处理
        expert_output = torch.zeros_like(x)
        for i, expert in enumerate(self.local_experts):
            local_expert_id = i + self.rank * (config['num_experts'] // self.world_size)
            expert_output += expert(x) * gate_probs[:, local_expert_id].unsqueeze(-1)
        
        # 跨GPU同步专家输出
        if dconfig['expert_parallel']:
            dist.all_reduce(expert_output, op=dist.ReduceOp.SUM)
        
        return expert_output

class DistributedPiKVManager:
    def __init__(self):
        self.world_size = dconfig['world_size']
        self.rank = dconfig['rank']
        self.device = dconfig['device']
        
        # 初始化分布式环境
        if not dist.is_initialized():
            dist.init_process_group(
                backend=dconfig['dist_backend'],
                init_method=dconfig['dist_url'],
                world_size=self.world_size,
                rank=self.rank
            )
        
        # 创建模型
        self.model = DistributedPiKVMoE().to(self.device)
        if dconfig['expert_parallel']:
            self.model = DDP(self.model, device_ids=[self.rank])
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        # 混合精度训练 - 移到这里，因为这是训练管理器的一部分
        self.scaler = GradScaler() if self.model.use_mixed_precision else None
        
    def train_step(self, data, target):
        self.model.train()
        self.optimizer.zero_grad()
        
        # 使用混合精度训练
        if dconfig['use_mixed_precision'] and self.scaler is not None:
            with autocast():
                output = self.model(data)
                loss = F.mse_loss(output, target)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def save_checkpoint(self, path):
        if self.rank == 0:  # 只在主进程保存检查点
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            }
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict']) 