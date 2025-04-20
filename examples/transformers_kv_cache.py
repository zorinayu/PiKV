import torch
import torch.nn as nn
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from core.single.pikv_moe import PiKVMoE
from core.single.config import config
from typing import Optional, Tuple, List, Union

class PiKVCache:
    def __init__(self, model_name: str = "gpt2", max_length: int = 1024):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Initialize PiKV MoE
        self.pikv = PiKVMoE(rank=4, alpha=1.0)
        
        # Initialize KV cache
        self.kv_cache = {}
        self.current_length = 0
    
    def _update_cache(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """Update KV cache for a specific layer."""
        if layer_idx not in self.kv_cache:
            self.kv_cache[layer_idx] = {
                'key': torch.zeros(self.max_length, key.size(-1), device=key.device),
                'value': torch.zeros(self.max_length, value.size(-1), device=value.device)
            }
        
        # Update cache
        self.kv_cache[layer_idx]['key'][self.current_length] = key
        self.kv_cache[layer_idx]['value'][self.current_length] = value
    
    def _get_cache(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get KV cache for a specific layer."""
        if layer_idx not in self.kv_cache:
            return None, None
        return self.kv_cache[layer_idx]['key'], self.kv_cache[layer_idx]['value']
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate text using PiKV cache."""
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        input_ids = input_ids.to(self.model.device)
        
        # Initialize output
        output_ids = input_ids.clone()
        self.current_length = input_ids.size(1)
        
        # Generate tokens
        for _ in range(max_new_tokens):
            # Get model outputs
            outputs = self.model(
                input_ids=output_ids,
                use_cache=True,
                return_dict=True
            )
            
            # Process each layer's KV cache
            for layer_idx, layer_output in enumerate(outputs.past_key_values):
                key, value = layer_output
                
                # Update PiKV cache
                self._update_cache(layer_idx, key, value)
                
                # Get cached values from PiKV
                cached_key, cached_value = self._get_cache(layer_idx)
                if cached_key is not None and cached_value is not None:
                    # Process through PiKV MoE
                    processed_key = self.pikv(cached_key)
                    processed_value = self.pikv(cached_value)
                    
                    # Update model's KV cache
                    outputs.past_key_values[layer_idx] = (processed_key, processed_value)
            
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

def main():
    # Initialize PiKV cache
    pikv_cache = PiKVCache(model_name="gpt2", max_length=1024)
    
    # Example prompts
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In a galaxy far far away"
    ]
    
    # Generate text for each prompt
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated_text = pikv_cache.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main() 