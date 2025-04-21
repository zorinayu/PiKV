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

class DistributedPiKVCache:
    def __init__(self, model_name: str = "gpt2", max_length: int = 1024):
        # Initialize distributed environment
        setup_distributed()
        
        # Get actual rank and local_rank from environment
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        print(f"Process started with rank {self.rank}, local_rank {self.local_rank}, world_size {self.world_size}")
        
        # Initialize model and tokenizer
        print(f"Rank {self.rank}: Loading model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Get model's hidden size
        self.hidden_size = self.model.config.hidden_size
        
        # Update global config with model's hidden size
        config['hidden_size'] = self.hidden_size
        
        # Initialize Distributed PiKV MoE with higher LoRA rank
        print(f"Rank {self.rank}: Initializing DistributedPiKVMoE...")
        self.pikv = DistributedPiKVMoE(rank=8, alpha=1.0)
        
        # Move model and PiKV to device
        self.device = torch.device(f"cuda:{self.local_rank}")
        print(f"Rank {self.rank}: Moving model to device {self.device}")
        self.model = self.model.to(self.device)
        self.pikv = self.pikv.to(self.device)
        
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
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate text using distributed PiKV cache."""
        try:
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            input_ids = input_ids.to(self.device)
            
            # Initialize output
            output_ids = input_ids.clone()
            self.current_length = input_ids.size(1)
            
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
                for layer_idx, layer_output in enumerate(outputs.past_key_values):
                    key, value = layer_output
                    
                    # Process through Distributed PiKV MoE
                    processed_key = self._process_with_pikv(key)
                    processed_value = self._process_with_pikv(value)
                    
                    # Create new tuple for this layer
                    new_past_key_values.append((processed_key, processed_value))
                
                # Update model's KV cache
                outputs.past_key_values = tuple(new_past_key_values)
                
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
            
            # Decode and return generated text
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Rank {self.rank}: Error in generate: {e}")
            raise

def main():
    try:
        # Initialize distributed PiKV cache
        print("Initializing DistributedPiKVCache...")
        pikv_cache = DistributedPiKVCache(model_name="gpt2", max_length=1024)
        
        # Example prompts
        prompts = [
            "The quick brown fox",
            "Once upon a time",
            "In a galaxy far far away"
        ]
        
        # Generate text for each prompt
        for prompt in prompts:
            if pikv_cache.rank == 0:  # Only print on main process
                print(f"\nPrompt: {prompt}")
            generated_text = pikv_cache.generate(
                prompt,
                max_new_tokens=50,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )
            if pikv_cache.rank == 0:  # Only print on main process
                print(f"Generated: {generated_text}")
        
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