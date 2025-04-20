import torch
import torch.nn as nn
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from core.single.pikv_moe import PiKVMoE
from core.single.config import config

class SimplestPiKVCache:
    def __init__(self, model_name: str = "gpt2", max_length: int = 1024):
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Get model's hidden size
        self.hidden_size = self.model.config.hidden_size
        
        # Update global config with model's hidden size
        config['hidden_size'] = self.hidden_size
        
        # Initialize PiKV MoE
        self.pikv = PiKVMoE(rank=4, alpha=1.0)
        
        # Initialize KV cache
        self.kv_cache = {}
        self.current_length = 0
    
    def _process_with_pikv(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process tensor through PiKV MoE."""
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
        
        # Ensure tensor has the correct hidden size
        if tensor.shape[-1] != self.hidden_size:
            # Project to hidden size if needed
            projection = nn.Linear(tensor.shape[-1], self.hidden_size).to(tensor.device)
            tensor = projection(tensor)
        
        # Process through PiKV MoE
        processed = self.pikv(tensor)
        
        # Reshape back to original dimensions if needed
        if len(tensor.shape) == 4:
            processed = processed.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        
        return processed
    
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
            new_past_key_values = []
            for layer_idx, layer_output in enumerate(outputs.past_key_values):
                key, value = layer_output
                
                # Process through PiKV MoE
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

def main():
    # Initialize PiKV cache
    pikv_cache = SimplestPiKVCache(model_name="gpt2", max_length=1024)
    
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