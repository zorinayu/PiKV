import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from .distributed_config import distributed_config as dconfig
from .config import config
from ..single.lora import LoRALayer, LoRAExpert, LoRAKVCache
from ..single.kv_cache_compression import KVCacheCompressor
from ..single.routing_strategy import AdaptiveRouter
import math

class DistributedExpert(nn.Module):
    def __init__(self, expert_id, world_size, rank=4, alpha=1.0):
        super(DistributedExpert, self).__init__()
        self.expert_id = expert_id
        self.world_size = world_size
        self.expert = LoRAExpert(config['hidden_size'], rank=rank, alpha=alpha)
        
    def forward(self, x):
        return self.expert(x)

class DistributedKVCache(nn.Module):
    def __init__(self, size, expert_id, world_size):
        super(DistributedKVCache, self).__init__()
        self.size = size
        self.hidden_size = config['hidden_size']
        self.expert_id = expert_id
        self.world_size = world_size
        
        # Initialize tensors
        self.register_buffer('keys', torch.zeros(size, self.hidden_size))
        self.register_buffer('values', torch.zeros(size, self.hidden_size))
        self.register_buffer('importance', torch.zeros(size))
        
        # Initialize compressor
        self.compressor = KVCacheCompressor(
            hidden_size=self.hidden_size,
            compression_type='pyramid',
            compression_ratio=0.5
        )
        
        # Initialize LoRA for cache values
        self.value_lora = LoRALayer(
            self.hidden_size,
            self.hidden_size,
            rank=4,
            alpha=1.0
        )
    
    def update(self, idx, key, value, importance):
        # Update cache at the specified index
        self.keys[idx] = key.mean(dim=0)  # Average across batch
        self.values[idx] = value.mean(dim=0)  # Average across batch
        self.importance[idx] = importance.mean().item()
    
    def get_all(self):
        # Apply compression to cached values
        compressed_keys, compressed_values = self.compressor(
            self.keys.unsqueeze(0),
            self.values.unsqueeze(0),
            self.importance.unsqueeze(0)
        )
        
        # Apply LoRA to compressed values
        compressed_values = compressed_values + self.value_lora(compressed_values)
        
        return compressed_values.squeeze(0).mean(dim=0)  # Return average of compressed values
    
    def set_all(self, data):
        if data is not None:
            self.values.copy_(data.unsqueeze(0).expand(self.size, -1))

class DistributedPiKVMoE(nn.Module):
    def __init__(self, rank=4, alpha=1.0):
        super(DistributedPiKVMoE, self).__init__()
        self.world_size = dconfig['world_size']
        self.rank = dconfig['rank']
        
        # Expert parallel: each GPU handles a subset of experts
        experts_per_gpu = config['num_experts'] // self.world_size
        self.local_experts = nn.ModuleList([
            DistributedExpert(i + self.rank * experts_per_gpu, self.world_size, rank=rank, alpha=alpha)
            for i in range(experts_per_gpu)
        ])
        
        # Initialize adaptive router with LoRA
        self.router = AdaptiveRouter(
            hidden_size=config['hidden_size'],
            num_experts=config['num_experts'],
            top_k=2,
            temperature=1.0
        )
        
        # Query-aware KV cache selection with LoRA
        self.query_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.key_proj = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.query_lora = LoRALayer(config['hidden_size'], config['hidden_size'], rank=rank, alpha=alpha)
        self.key_lora = LoRALayer(config['hidden_size'], config['hidden_size'], rank=rank, alpha=alpha)
        
        # Cache size allocation
        self.cache_sizes = self.pyramidal_cache_allocation()
        
        # Initialize KV caches for each local expert
        self.kv_caches = nn.ModuleList([
            DistributedKVCache(size, i + self.rank * experts_per_gpu, self.world_size)
            for i, size in enumerate(self.cache_sizes[:experts_per_gpu])
        ])
        
        # Cache pointers for each expert
        self.register_buffer('cache_ptrs', torch.zeros(experts_per_gpu, dtype=torch.long))
        
        # Mixed precision training
        self.use_mixed_precision = dconfig['use_mixed_precision']
        
    def pyramidal_cache_allocation(self):
        C1 = config['kv_cache_size']
        d = config['cache_decrement']
        return [C1 - (i - 1) * d for i in range(1, config['num_layers'] + 1)]
    
    def compute_token_importance(self, query, key):
        # Project query and key with LoRA
        query_base = self.query_proj(query)
        key_base = self.key_proj(key)
        
        query_lora = self.query_lora(query)
        key_lora = self.key_lora(key)
        
        query = query_base + query_lora
        key = key_base + key_lora
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(config['hidden_size'])
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Compute importance as sum of attention weights
        importance = attention_probs.sum(dim=1)  # [batch_size, seq_len]
        
        return importance
    
    def update_cache(self, expert_idx, key, value, importance):
        cache = self.kv_caches[expert_idx]
        ptr = self.cache_ptrs[expert_idx]
        
        # Update cache with new key-value pair
        cache.update(ptr, key, value, importance)
        
        # Update pointer
        self.cache_ptrs[expert_idx] = (ptr + 1) % cache.size
    
    def forward(self, x, query=None):
        # Calculate routing weights using adaptive router
        routing_weights, expert_indices, top_k_weights, lb_loss, importance = self.router(x)
        
        # If query is provided, compute token importance
        if query is not None:
            importance = self.compute_token_importance(query, x)
        
        # Initialize output tensor
        expert_output = torch.zeros_like(x)
        
        # Process each local expert
        for i, expert in enumerate(self.local_experts):
            # Get expert output with LoRA
            expert_output_i = expert(x)
            
            # Get cached values
            cached_values = self.kv_caches[i].get_all()
            
            # Combine with cached values
            if cached_values is not None:
                expert_output_i = expert_output_i + cached_values.detach()
            
            # Update cache with new values
            self.update_cache(i, x.detach(), expert_output_i.detach(), importance.detach())
            
            # Add to final output weighted by routing probabilities
            local_expert_id = i + self.rank * (config['num_experts'] // self.world_size)
            expert_output += expert_output_i * routing_weights[:, :, local_expert_id].unsqueeze(-1)
        
        # Synchronize expert outputs across GPUs
        if dconfig['expert_parallel']:
            dist.all_reduce(expert_output, op=dist.ReduceOp.SUM)
        
        return expert_output, lb_loss

class DistributedPiKVManager:
    def __init__(self, rank=4, alpha=1.0):
        self.world_size = dconfig['world_size']
        self.rank = dconfig['rank']
        self.device = dconfig['device']
        
        # Initialize distributed environment
        if not dist.is_initialized():
            dist.init_process_group(
                backend=dconfig['dist_backend'],
                init_method=dconfig['dist_url'],
                world_size=self.world_size,
                rank=self.rank
            )
        
        # Create model
        self.model = DistributedPiKVMoE(rank=rank, alpha=alpha).to(self.device)
        
        # Wrap model with DDP if using expert parallel
        if dconfig['expert_parallel']:
            self.model = DDP(self.model, device_ids=[self.rank])
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        # Mixed precision training
        self.scaler = GradScaler() if self.model.use_mixed_precision else None
        
    def train_step(self, data, target):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Use mixed precision training
        if dconfig['use_mixed_precision'] and self.scaler is not None:
            with autocast():
                output, lb_loss = self.model(data)
                loss = F.mse_loss(output, target) + lb_loss
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output, lb_loss = self.model(data)
            loss = F.mse_loss(output, target) + lb_loss
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def save_checkpoint(self, path):
        if self.rank == 0:  # Only save checkpoint on main process
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