import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config
from .kv_cache_compression import KVCacheCompressor
from .routing_strategy import AdaptiveRouter
from .lora import LoRALayer, LoRAExpert, LoRAKVCache
from .distillation import PiKVDistillation, create_teacher_model, distillation_training_step
from .shared import ExternalMemoryCache
import math

class KVCache(nn.Module):
    """
    KV Cache implementation with compression and streaming support.
    """
    def __init__(self, size):
        super(KVCache, self).__init__()
        self.size = size
        self.hidden_size = config['hidden_size']
        
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
        # Reshape input if needed
        if len(key.shape) == 3:  # [batch_size, seq_len, hidden_size]
            key = key.mean(dim=0).mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0).mean(dim=0)  # [hidden_size]
        elif len(key.shape) == 2:  # [seq_len, hidden_size]
            key = key.mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0)  # [hidden_size]
        
        # Update cache at the specified index
        self.keys[idx] = key
        self.values[idx] = value
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

class PiKVMoE(nn.Module):
    def __init__(self, rank=4, alpha=1.0, use_distillation=False, teacher_hidden_size=None):
        super(PiKVMoE, self).__init__()
        self.use_distillation = use_distillation
        
        self.experts = nn.ModuleList([
            LoRAExpert(config['hidden_size'], rank=rank, alpha=alpha)
            for _ in range(config['num_experts'])
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
        
        # Cache size allocation for each layer
        self.cache_sizes = self.pyramidal_cache_allocation()
        
        # Initialize KV caches for each expert
        self.kv_caches = nn.ModuleList([
            KVCache(size) for size in self.cache_sizes
        ])
        
        # Cache pointers for each expert
        self.register_buffer('cache_ptrs', torch.zeros(config['num_experts'], dtype=torch.long))
        
        # Projection to vocabulary size
        self.vocab_proj = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        # Memory expansion flag
        self.use_memory_expansion = config.get('use_memory_expansion', False)
        if self.use_memory_expansion:
            self.external_cache = ExternalMemoryCache()
        
        # Knowledge Distillation Setup
        if use_distillation:
            self.teacher_hidden_size = teacher_hidden_size or config['hidden_size'] * 2
            self.distillation_module = PiKVDistillation(
                student_hidden_size=config['hidden_size'],
                teacher_hidden_size=self.teacher_hidden_size,
                num_experts=config['num_experts'],
                temperature=4.0,
                expert_distill_weight=0.4,
                cache_distill_weight=0.3
            )
            
            # Create teacher model (can be loaded from checkpoint)
            self.teacher_model = create_teacher_model(
                hidden_size=self.teacher_hidden_size,
                num_experts=config['num_experts']
            )
            
            # Freeze teacher model parameters
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            print(f"Knowledge Distillation enabled with teacher hidden size: {self.teacher_hidden_size}")

    def pyramidal_cache_allocation(self):
        """
        Calculate the cache size for each layer using the pyramidal allocation policy.
        """
        C1 = config['kv_cache_size']
        d = config['cache_decrement']
        return [C1 - (i - 1) * d for i in range(1, config['num_layers'] + 1)]
    
    def compute_token_importance(self, query, key):
        """
        Compute importance scores for tokens based on query-key attention.
        """
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
        """
        Update KV cache for a specific expert with new key-value pairs.
        """
        cache = self.kv_caches[expert_idx]
        ptr = self.cache_ptrs[expert_idx]
        
        # Update cache with new key-value pair
        cache.update(ptr, key, value, importance)
        
        # Update pointer
        self.cache_ptrs[expert_idx] = (ptr + 1) % cache.size
    
    def forward(self, x, query=None, return_loss=False, targets=None, use_teacher=False):
        # Convert input to float if it's not already
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        
        # Calculate routing weights using adaptive router
        routing_weights, expert_indices, top_k_weights, lb_loss, importance = self.router(x)
        
        # If query is provided, compute token importance
        if query is not None:
            if query.dtype != torch.float32:
                query = query.to(torch.float32)
            importance = self.compute_token_importance(query, x)
        
        # Initialize output tensor
        batch_size, seq_len, hidden_size = x.shape
        expert_output = torch.zeros(batch_size, seq_len, hidden_size, device=x.device)
        
        # Collect expert outputs for distillation
        expert_outputs_list = []
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Get expert output with LoRA
            expert_output_i = expert(x)  # [batch_size, seq_len, hidden_size]
            expert_outputs_list.append(expert_output_i)
            
            # Get cached values
            cached_values = self.kv_caches[i].get_all()
            
            # Combine with cached values
            if cached_values is not None:
                expert_output_i = expert_output_i + cached_values.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            
            # Update cache with new values
            self.update_cache(i, x.detach(), expert_output_i.detach(), importance.detach())
            
            # Add to final output weighted by routing probabilities
            expert_output += expert_output_i * routing_weights[:, :, i].unsqueeze(-1)
        
        # Project to vocabulary size if needed
        logits = self.vocab_proj(expert_output)
        
        # Knowledge Distillation
        distill_loss = torch.tensor(0.0, device=x.device)
        if self.use_distillation and self.training and use_teacher:
            # Get teacher outputs
            with torch.no_grad():
                teacher_outputs = self.teacher_model(x)
            
            # Compute distillation loss
            distill_loss, distill_loss_dict = self.distillation_module(
                student_logits=logits,
                teacher_logits=teacher_outputs['logits'],
                student_features=expert_output,
                teacher_features=teacher_outputs['features'],
                student_expert_outputs=expert_outputs_list,
                teacher_expert_outputs=teacher_outputs.get('expert_outputs'),
                teacher_routing_weights=teacher_outputs.get('routing_weights'),
                targets=targets
            )
        
        if return_loss:
            total_loss = lb_loss + distill_loss
            return logits, total_loss
        return logits
    
    def enable_distillation(self, teacher_model_path=None):
        """
        Enable knowledge distillation and optionally load teacher model
        """
        if not self.use_distillation:
            print("Distillation not initialized. Please set use_distillation=True during initialization.")
            return
        
        if teacher_model_path:
            self.load_teacher_model(teacher_model_path)
        
        print("Knowledge distillation enabled")
    
    def disable_distillation(self):
        """
        Disable knowledge distillation for inference
        """
        self.use_distillation = False
        print("Knowledge distillation disabled")
    
    def load_teacher_model(self, model_path):
        """
        Load pre-trained teacher model
        """
        if not self.use_distillation:
            print("Distillation not initialized")
            return
        
        try:
            checkpoint = torch.load(model_path, map_location=next(self.parameters()).device)
            self.teacher_model.load_state_dict(checkpoint)
            print(f"Teacher model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load teacher model: {e}")
    
    def distillation_step(self, input_data, targets=None, optimizer=None):
        """
        Perform a single distillation training step
        """
        if not self.use_distillation:
            print("Distillation not enabled")
            return {}
        
        return distillation_training_step(
            student_model=self,
            teacher_model=self.teacher_model,
            distillation_module=self.distillation_module,
            input_data=input_data,
            targets=targets,
            optimizer=optimizer
        )
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint with KV caches and LoRA parameters.
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'cache_ptrs': self.cache_ptrs,
            'kv_caches': [cache.get_all() for cache in self.kv_caches],
            'use_distillation': self.use_distillation
        }
        
        # Save teacher model if distillation is enabled
        if self.use_distillation:
            checkpoint['teacher_state_dict'] = self.teacher_model.state_dict()
            checkpoint['distillation_state_dict'] = self.distillation_module.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint with KV caches and LoRA parameters.
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.cache_ptrs.copy_(checkpoint['cache_ptrs'])
        for i, cache_data in enumerate(checkpoint['kv_caches']):
            if cache_data is not None:
                self.kv_caches[i].set_all(cache_data)
        
        # Load distillation components if available
        if checkpoint.get('use_distillation', False) and self.use_distillation:
            if 'teacher_state_dict' in checkpoint:
                self.teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
            if 'distillation_state_dict' in checkpoint:
                self.distillation_module.load_state_dict(checkpoint['distillation_state_dict'])
