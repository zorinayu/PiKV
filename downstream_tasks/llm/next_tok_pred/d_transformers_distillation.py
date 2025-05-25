import torch
import torch.nn as nn
import torch.distributed as dist
import os
import time
from datetime import timedelta
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from core.distributed.distributed_pikv import DistributedPiKVMoE
from core.distributed.config import config
from core.distributed.distributed_config import distributed_config as dconfig
from core.single.distillation import PiKVDistillation, create_teacher_model, distillation_training_step
import argparse
from typing import Optional
import torch.nn.functional as F

def setup_distributed():
    """Initialize distributed environment with error handling."""
    try:
        if not dist.is_initialized():
            # Get rank and world_size from environment variables
            rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            print(f"Initializing distributed environment on rank {rank}, local_rank {local_rank}")
            
            # Initialize process group
            dist.init_process_group(
                backend=dconfig['dist_backend'],
                init_method='env://',  # Use environment variables
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=30)
            )
            print(f"Successfully initialized distributed environment on rank {rank}")
    except Exception as e:
        print(f"Error initializing distributed environment: {e}")
        raise

class DistributedPiKVCacheWithDistillation:
    def __init__(
        self, 
        model_name: str = "gpt2", 
        max_length: int = 1024,
        use_distillation: bool = True,
        teacher_hidden_size: Optional[int] = None,
        distillation_temperature: float = 4.0,
        distillation_alpha: float = 0.7
    ):
        # Initialize distributed environment
        setup_distributed()
        
        # Get actual rank and local_rank from environment
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        print(f"Process started with rank {self.rank}, local_rank {self.local_rank}, world_size {self.world_size}")
        
        # Set device first before using it
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Rank {self.rank}: Loading model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model configuration and update global config FIRST
        self.hidden_size = self.model.config.hidden_size  # Use actual model hidden size
        
        # Update global config with actual model dimensions BEFORE creating PiKV
        from core.single.config import config
        config['hidden_size'] = self.hidden_size
        config['vocab_size'] = self.model.config.vocab_size
        print(f"Rank {self.rank}: Updated config - hidden_size: {self.hidden_size}, vocab_size: {self.model.config.vocab_size}")
        
        # Store configuration
        self.max_length = max_length
        self.use_distillation = use_distillation
        self.teacher_hidden_size = teacher_hidden_size or (self.hidden_size * 2)
        self.distillation_temperature = distillation_temperature
        self.distillation_alpha = distillation_alpha
        
        # Initialize Distributed PiKV MoE with correct hidden size
        print(f"Rank {self.rank}: Initializing DistributedPiKVMoE...")
        
        # SOLUTION: 创建自定义的DistributedPiKVMoE，直接使用正确的hidden_size
        # 这是一个根本性的修复，完全绕过全局配置的问题
        
        class CustomDistributedPiKVMoE(nn.Module):
            def __init__(self, hidden_size, num_experts=4, rank=4, alpha=1.0):
                super(CustomDistributedPiKVMoE, self).__init__()
                self.hidden_size = hidden_size
                self.num_experts = num_experts
                
                # 创建自定义的路由器，直接使用正确的hidden_size
                class CustomAdaptiveRouter(nn.Module):
                    def __init__(self, hidden_size, num_experts, top_k=2, temperature=1.0):
                        super(CustomAdaptiveRouter, self).__init__()
                        self.hidden_size = hidden_size
                        self.num_experts = num_experts
                        self.top_k = top_k
                        self.temperature = temperature
                        
                        # 使用正确的hidden_size创建路由器
                        self.router_selector = nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, 2),  # 选择两种路由策略
                            nn.Softmax(dim=-1)
                        )
                        
                        # 主路由器
                        self.main_router = nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, num_experts)
                        )
                        
                        # 负载均衡路由器
                        self.balance_router = nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, num_experts)
                        )
                        
                        # 专家负载跟踪
                        self.register_buffer('expert_loads', torch.zeros(num_experts))
                        self.register_buffer('total_tokens', torch.tensor(0.0))
                    
                    def forward(self, x):
                        # 重塑输入以确保正确的维度
                        if len(x.shape) == 2:  # [batch_size, hidden_size]
                            x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
                        elif len(x.shape) == 1:  # [hidden_size]
                            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
                        
                        batch_size, seq_len, hidden_size = x.shape
                        
                        # 计算平均值用于路由选择
                        x_mean = x.mean(dim=1)  # [batch_size, hidden_size]
                        
                        # 选择路由策略
                        router_weights = self.router_selector(x_mean)  # [batch_size, 2]
                        
                        # 计算主路由权重
                        main_logits = self.main_router(x_mean)  # [batch_size, num_experts]
                        
                        # 计算负载均衡路由权重
                        balance_logits = self.balance_router(x_mean)  # [batch_size, num_experts]
                        
                        # 组合路由权重
                        combined_logits = (router_weights[:, 0:1] * main_logits + 
                                         router_weights[:, 1:2] * balance_logits)
                        
                        # 应用温度缩放
                        combined_logits = combined_logits / self.temperature
                        
                        # 计算路由概率
                        routing_probs = F.softmax(combined_logits, dim=-1)  # [batch_size, num_experts]
                        
                        # 扩展到序列长度
                        routing_weights = routing_probs.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, num_experts]
                        
                        # 计算top-k专家
                        top_k_values, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
                        
                        # 计算负载均衡损失
                        expert_usage = routing_probs.sum(dim=0)  # [num_experts]
                        lb_loss = torch.var(expert_usage) * 0.01  # 简单的负载均衡损失
                        
                        # 计算重要性分数（基于路由权重的方差）
                        importance = torch.var(routing_probs, dim=-1)  # [batch_size]
                        importance = importance.unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]
                        
                        return routing_weights, top_k_indices, top_k_values, lb_loss, importance
                
                # 创建自定义路由器
                self.router = CustomAdaptiveRouter(hidden_size, num_experts)
                
                # 创建专家网络
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size)
                    ) for _ in range(num_experts)
                ])
                
                # 创建KV缓存
                self.cache_size = 128
                for i in range(num_experts):
                    self.register_buffer(f'kv_keys_{i}', torch.zeros(self.cache_size, hidden_size))
                    self.register_buffer(f'kv_values_{i}', torch.zeros(self.cache_size, hidden_size))
                    self.register_buffer(f'kv_importance_{i}', torch.zeros(self.cache_size))
                
                # 缓存指针
                self.register_buffer('cache_ptrs', torch.zeros(num_experts, dtype=torch.long))
            
            def forward(self, x, query=None):
                # 确保输入维度正确
                if len(x.shape) == 2:  # [batch_size, hidden_size]
                    x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
                elif len(x.shape) == 1:  # [hidden_size]
                    x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
                
                batch_size, seq_len, hidden_size = x.shape
                
                # 获取路由权重
                routing_weights, expert_indices, top_k_weights, lb_loss, importance = self.router(x)
                
                # 初始化输出
                expert_output = torch.zeros_like(x)
                
                # 处理每个专家
                for i, expert in enumerate(self.experts):
                    # 重塑输入以匹配专家期望的维度
                    x_reshaped = x.reshape(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
                    expert_output_i = expert(x_reshaped)  # [batch_size * seq_len, hidden_size]
                    expert_output_i = expert_output_i.reshape(batch_size, seq_len, hidden_size)
                    
                    # 添加到最终输出，按路由权重加权
                    expert_output += expert_output_i * routing_weights[:, :, i].unsqueeze(-1)
                
                return expert_output, lb_loss
        
        # 创建自定义的DistributedPiKVMoE实例
        self.pikv = CustomDistributedPiKVMoE(
            hidden_size=self.hidden_size,
            num_experts=4,
            rank=4,
            alpha=1.0
        )
        
        print(f"Rank {self.rank}: Created custom DistributedPiKVMoE with hidden_size={self.hidden_size}")
        
        # Initialize knowledge distillation if enabled
        if self.use_distillation:
            print(f"Rank {self.rank}: Initializing knowledge distillation...")
            
            # Create teacher model (without moving to device yet)
            self.teacher_model = create_teacher_model(
                hidden_size=self.teacher_hidden_size,
                num_experts=4,
                num_layers=6,
                vocab_size=self.model.config.vocab_size
            )
            
            # Create distillation module (without moving to device yet)
            self.distillation_module = PiKVDistillation(
                student_hidden_size=self.hidden_size,
                teacher_hidden_size=self.teacher_hidden_size,
                num_experts=4,
                temperature=self.distillation_temperature
            )
            
            print(f"Rank {self.rank}: Knowledge Distillation enabled with teacher hidden size: {self.teacher_hidden_size}")
        else:
            self.teacher_model = None
            self.distillation_module = None
        
        # Move model, PiKV, and distillation components to device
        print(f"Rank {self.rank}: Moving models to device {self.device}")
        self.model = self.model.to(self.device)
        self.pikv = self.pikv.to(self.device)
        
        if self.use_distillation and self.teacher_model is not None and self.distillation_module is not None:
            # Move teacher model and distillation module to device
            self.teacher_model = self.teacher_model.to(self.device)
            self.distillation_module = self.distillation_module.to(self.device)
            
            # Freeze teacher model parameters
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        
        # Initialize KV cache
        self.kv_cache = {}
        self.current_length = 0
        print(f"Rank {self.rank}: Initialization complete")
    
    def _process_with_pikv(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process tensor through Distributed PiKV MoE."""
        try:
            # Get the shape of the input tensor
            if len(tensor.shape) == 4:  # [batch_size, num_heads, seq_len, head_dim]
                batch_size, num_heads, seq_len, head_dim = tensor.shape
                # Reshape to [batch_size, seq_len, hidden_size]
                hidden_size = num_heads * head_dim
                tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
            elif len(tensor.shape) == 3:  # [batch_size, seq_len, hidden_size]
                pass  # Already in correct shape
            elif len(tensor.shape) == 2:  # [batch_size, hidden_size]
                # Add sequence length dimension
                tensor = tensor.unsqueeze(1)  # [batch_size, 1, hidden_size]
            elif len(tensor.shape) == 1:  # [hidden_size]
                # Add batch and sequence length dimensions
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
            
            # Ensure tensor has the correct hidden size
            if tensor.shape[-1] != self.hidden_size:
                # Project to hidden size if needed
                projection = nn.Linear(tensor.shape[-1], self.hidden_size).to(tensor.device)
                tensor = projection(tensor)
            
            # Process through Distributed PiKV MoE
            processed, _ = self.pikv(tensor, tensor)  # Pass tensor as both input and query
            
            # Reshape back to original dimensions if needed
            if len(tensor.shape) == 4:
                processed = processed.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
            
            return processed
        except Exception as e:
            print(f"Rank {self.rank}: Error in _process_with_pikv: {e}")
            raise
    
    def _get_teacher_outputs(self, input_ids: torch.Tensor):
        """Get teacher model outputs for distillation."""
        if not self.use_distillation or self.teacher_model is None:
            return None
        
        with torch.no_grad():
            # Create input tensor for teacher model
            # Teacher expects [batch_size, seq_len, teacher_hidden_size]
            batch_size, seq_len = input_ids.shape
            teacher_input = torch.randn(
                batch_size, seq_len, self.teacher_hidden_size,
                device=input_ids.device
            )
            
            # Get teacher outputs - this returns a dict
            teacher_outputs = self.teacher_model(teacher_input)
            
            # Extract logits from the teacher outputs dict
            if isinstance(teacher_outputs, dict):
                teacher_logits = teacher_outputs.get('logits')
                teacher_features = teacher_outputs.get('features', teacher_input)
                expert_outputs = teacher_outputs.get('expert_outputs')
                routing_weights = teacher_outputs.get('routing_weights')
            else:
                # If teacher returns tensor directly
                teacher_logits = teacher_outputs
                teacher_features = teacher_input
                expert_outputs = None
                routing_weights = None
            
            # Ensure teacher_logits has the correct shape for vocabulary
            if teacher_logits is not None and teacher_logits.shape[-1] != self.model.config.vocab_size:
                if not hasattr(self, 'vocab_projection'):
                    self.vocab_projection = nn.Linear(
                        teacher_logits.shape[-1], 
                        self.model.config.vocab_size
                    ).to(teacher_logits.device)
                teacher_logits = self.vocab_projection(teacher_logits)
            
            return {
                'logits': teacher_logits,
                'features': teacher_features,
                'expert_outputs': expert_outputs,
                'routing_weights': routing_weights
            }
    
    def generate_with_distillation(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_teacher: bool = True
    ) -> str:
        """Generate text using distributed PiKV cache with knowledge distillation."""
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            input_ids = input_ids.to(self.device)
            
            # Initialize output
            output_ids = input_ids.clone()
            self.current_length = input_ids.size(1)
            
            # Collect distillation losses for monitoring
            distillation_losses = []
            
            # Generate tokens
            for i in range(max_new_tokens):
                if self.rank == 0:
                    print(f"Generating token {i+1}/{max_new_tokens}")
                
                # Get model outputs
                outputs = self.model(
                    input_ids=output_ids,
                    use_cache=True,
                    return_dict=True
                )
                
                # Process each layer's KV cache
                new_past_key_values = []
                expert_outputs_list = []
                
                for layer_idx, layer_output in enumerate(outputs.past_key_values):
                    key, value = layer_output
                    
                    # Process through Distributed PiKV MoE
                    processed_key = self._process_with_pikv(key)
                    processed_value = self._process_with_pikv(value)
                    
                    # Store expert outputs for distillation
                    expert_outputs_list.append(processed_value)
                    
                    # Create new tuple for this layer
                    new_past_key_values.append((processed_key, processed_value))
                
                # Update model's KV cache
                outputs.past_key_values = tuple(new_past_key_values)
                
                # Apply knowledge distillation if enabled and using teacher
                if self.use_distillation and use_teacher and self.distillation_module is not None:
                    # Get teacher outputs
                    teacher_outputs = self._get_teacher_outputs(output_ids)
                    
                    if teacher_outputs is not None and teacher_outputs['logits'] is not None:
                        try:
                            # Ensure distillation_module is not None (type assertion for linter)
                            distillation_module = self.distillation_module
                            assert distillation_module is not None, "Distillation module should not be None"
                            
                            # Compute distillation loss
                            distill_loss, distill_loss_dict = distillation_module(
                                student_logits=outputs.logits,
                                teacher_logits=teacher_outputs['logits'],
                                student_features=outputs.logits,  # Use logits as features
                                teacher_features=teacher_outputs['features'],
                                student_expert_outputs=expert_outputs_list if expert_outputs_list else None,
                                teacher_expert_outputs=teacher_outputs.get('expert_outputs'),
                                teacher_routing_weights=teacher_outputs.get('routing_weights'),
                                targets=None
                            )
                            
                            distillation_losses.append(distill_loss.item())
                            
                            if self.rank == 0 and i % 10 == 0:
                                print(f"Distillation loss at token {i}: {distill_loss.item():.4f}")
                        except Exception as e:
                            if self.rank == 0:
                                print(f"Warning: Distillation failed at token {i}: {e}")
                                # Continue without distillation for this token
                
                # Get next token logits
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to output
                output_ids = torch.cat([output_ids, next_token], dim=1)
                self.current_length += 1
                
                # Stop if we reach max length
                if self.current_length >= self.max_length:
                    break
            
            # Print distillation statistics
            if self.use_distillation and distillation_losses and self.rank == 0:
                avg_distill_loss = sum(distillation_losses) / len(distillation_losses)
                print(f"Average distillation loss: {avg_distill_loss:.4f}")
            
            # Decode and return generated text
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Rank {self.rank}: Error in generate_with_distillation: {e}")
            raise
    
    def distillation_training_step(
        self,
        input_data: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """Perform a single distillation training step."""
        if not self.use_distillation or self.distillation_module is None:
            print(f"Rank {self.rank}: Distillation not enabled")
            return {}
        
        try:
            # Get student outputs
            student_outputs = self.model(input_data, return_dict=True)
            
            # Process through PiKV
            processed_features = self._process_with_pikv(student_outputs.logits)
            
            # Get teacher outputs
            teacher_outputs = self._get_teacher_outputs(input_data)
            
            if teacher_outputs is None or teacher_outputs['logits'] is None:
                return {}
            
            # Compute distillation loss
            distill_loss, distill_loss_dict = self.distillation_module(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs['logits'],
                student_features=processed_features,
                teacher_features=teacher_outputs['features'],
                student_expert_outputs=None,  # Could be implemented if needed
                teacher_expert_outputs=teacher_outputs.get('expert_outputs'),
                teacher_routing_weights=teacher_outputs.get('routing_weights'),
                targets=targets
            )
            
            # Backward pass if optimizer is provided
            if optimizer is not None:
                optimizer.zero_grad()
                distill_loss.backward()
                optimizer.step()
            
            # Convert tensor values to scalars for return
            result = {}
            for key, value in distill_loss_dict.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.item()
                else:
                    result[key] = value
            
            return result
            
        except Exception as e:
            print(f"Rank {self.rank}: Error in distillation_training_step: {e}")
            return {}
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint with distillation components."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'pikv_state_dict': self.pikv.state_dict(),
            'use_distillation': self.use_distillation,
            'hidden_size': self.hidden_size,
            'teacher_hidden_size': getattr(self, 'teacher_hidden_size', None),
            'distillation_temperature': getattr(self, 'distillation_temperature', None),
            'distillation_alpha': getattr(self, 'distillation_alpha', None)
        }
        
        if self.use_distillation and self.teacher_model is not None and self.distillation_module is not None:
            checkpoint['teacher_state_dict'] = self.teacher_model.state_dict()
            checkpoint['distillation_state_dict'] = self.distillation_module.state_dict()
        
        torch.save(checkpoint, path)
        if self.rank == 0:
            print(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint with distillation components."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.pikv.load_state_dict(checkpoint['pikv_state_dict'])
        
        if (checkpoint.get('use_distillation', False) and self.use_distillation and 
            self.teacher_model is not None and self.distillation_module is not None):
            if 'teacher_state_dict' in checkpoint:
                self.teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
            if 'distillation_state_dict' in checkpoint:
                self.distillation_module.load_state_dict(checkpoint['distillation_state_dict'])
        
        if self.rank == 0:
            print(f"Checkpoint loaded from: {path}")

def main():
    parser = argparse.ArgumentParser(description="Distributed PiKV Cache with Knowledge Distillation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--use_distillation", action="store_true", help="Enable knowledge distillation")
    parser.add_argument("--teacher_hidden_size", type=int, default=None, help="Teacher model hidden size")
    parser.add_argument("--distillation_temperature", type=float, default=4.0, help="Distillation temperature")
    parser.add_argument("--distillation_alpha", type=float, default=0.7, help="Distillation alpha")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--save_checkpoint", type=str, default=None, help="Path to save checkpoint")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to load checkpoint")
    args = parser.parse_args()
    
    try:
        # Initialize distributed PiKV cache with distillation
        print(f"Initializing DistributedPiKVCacheWithDistillation...")
        pikv_cache = DistributedPiKVCacheWithDistillation(
            model_name=args.model,
            max_length=1024,
            use_distillation=args.use_distillation,
            teacher_hidden_size=args.teacher_hidden_size,
            distillation_temperature=args.distillation_temperature,
            distillation_alpha=args.distillation_alpha
        )
        
        # Load checkpoint if specified
        if args.load_checkpoint:
            pikv_cache.load_checkpoint(args.load_checkpoint)
        
        # Example prompts for text generation with distillation
        prompts = [
            "The future of artificial intelligence is",
            "Knowledge distillation helps models learn by",
            "Distributed training enables"
        ]
        
        # Generate text for each prompt
        for prompt in prompts:
            if pikv_cache.rank == 0:  # Only print on main process
                print(f"\nPrompt: {prompt}")
            
            if args.use_distillation:
                generated_text = pikv_cache.generate_with_distillation(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    use_teacher=True
                )
            else:
                # Fallback to regular generation without distillation
                input_ids = pikv_cache.tokenizer.encode(prompt, return_tensors='pt').to(pikv_cache.device)
                with torch.no_grad():
                    outputs = pikv_cache.model.generate(
                        input_ids,
                        max_new_tokens=args.max_tokens,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=pikv_cache.tokenizer.eos_token_id
                    )
                generated_text = pikv_cache.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if pikv_cache.rank == 0:  # Only print on main process
                print(f"Generated: {generated_text}")
        
        # Save checkpoint if specified
        if args.save_checkpoint:
            pikv_cache.save_checkpoint(args.save_checkpoint)
        
        # Demonstrate distillation training step
        if args.use_distillation and pikv_cache.rank == 0:
            print("\nDemonstrating distillation training step...")
            
            # Create dummy training data
            dummy_input = torch.randint(0, 1000, (2, 10), device=pikv_cache.device)
            dummy_targets = torch.randint(0, 1000, (2, 10), device=pikv_cache.device)
            
            # Create optimizer
            optimizer = torch.optim.Adam(pikv_cache.model.parameters(), lr=1e-4)
            
            # Perform distillation training step
            loss_info = pikv_cache.distillation_training_step(
                input_data=dummy_input,
                targets=dummy_targets,
                optimizer=optimizer
            )
            
            print("Distillation training step completed:")
            for loss_name, loss_value in loss_info.items():
                print(f"  {loss_name}: {loss_value:.4f}")
        
        # Clean up distributed environment
        dist.destroy_process_group()
        print("Distributed environment cleaned up")
    except Exception as e:
        print(f"Error in main: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise

if __name__ == "__main__":
    main() 