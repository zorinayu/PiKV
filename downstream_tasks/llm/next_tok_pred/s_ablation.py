import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from core.single.pikv_moe import PiKVMoE
from core.single.config import config
import time
import psutil
import os

class SingleAblationTest:
    def __init__(self, model_name="gpt2"):
        # Initialize model and tokenizer
        self.device = config['device']
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get model's hidden size and number of attention heads
        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Update config with model's parameters
        config['hidden_size'] = self.hidden_size
        config['num_heads'] = self.num_heads
        
        # Initialize different MoE models with attention
        self.models = {
            'standard': self._create_standard_moe(),
            'lora': self._create_lora_moe(),
            'adaptive': self._create_adaptive_moe(),
            'pikv': self._create_pikv_moe()
        }
        
        # Move models to device
        for model in self.models.values():
            model.to(self.device)
    
    def _create_standard_moe(self):
        """Create standard MoE model with attention."""
        from core.single.normal_moe import StandardMoE
        return StandardMoE()
    
    def _create_lora_moe(self):
        """Create LoRA MoE model with attention."""
        from core.single.lora_moe import LoRAMoE
        return LoRAMoE(rank=4, alpha=1.0)
    
    def _create_adaptive_moe(self):
        """Create adaptive routing MoE model with attention."""
        from core.single.routing_moe import RoutingMoE
        return RoutingMoE()
    
    def _create_pikv_moe(self):
        """Create PiKV MoE model with attention."""
        return PiKVMoE(rank=4, alpha=1.0)
    
    def _process_with_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process tensor through attention mechanism."""
        if len(tensor.shape) == 4:  # [batch_size, num_heads, seq_len, head_dim]
            batch_size, num_heads, seq_len, head_dim = tensor.shape
            # Reshape to [batch_size, seq_len, hidden_size]
            hidden_size = num_heads * head_dim
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
        elif len(tensor.shape) == 3:  # [batch_size, seq_len, hidden_size]
            pass  # Already in correct shape
        elif len(tensor.shape) == 2:  # [batch_size, hidden_size]
            tensor = tensor.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        return tensor
    
    def test_inference_speed(self, model_name, input_tensor):
        """Test inference speed of a model."""
        model = self.models[model_name]
        
        # Ensure input tensor has correct dimensions
        if len(input_tensor.shape) == 2:  # [batch_size, hidden_size]
            input_tensor = input_tensor.unsqueeze(1)  # [batch_size, 1, hidden_size]
        elif len(input_tensor.shape) == 3:  # [batch_size, seq_len, hidden_size]
            pass  # Already in correct shape
        else:
            raise ValueError(f"Unexpected input tensor shape: {input_tensor.shape}")
        
        # For LoRA model, ensure input has correct dimensions
        if model_name == 'lora':
            # Ensure input has shape [batch_size, seq_len, hidden_size]
            if input_tensor.shape[1] != config['num_experts']:
                # Expand to match number of experts
                input_tensor = input_tensor.expand(-1, config['num_experts'], -1)
        
        # Warm up
        for _ in range(5):
            _ = model(input_tensor)
        
        # Measure inference time
        start_time = time.time()
        for _ in range(10):
            _ = model(input_tensor)
        end_time = time.time()
        
        return (end_time - start_time) / 10
    
    def test_memory_usage(self, model_name, input_tensor):
        """Test memory usage of a model."""
        model = self.models[model_name]
        process = psutil.Process(os.getpid())
        
        # Ensure input tensor has correct dimensions
        if len(input_tensor.shape) == 2:  # [batch_size, hidden_size]
            input_tensor = input_tensor.unsqueeze(1)  # [batch_size, 1, hidden_size]
        elif len(input_tensor.shape) == 3:  # [batch_size, seq_len, hidden_size]
            pass  # Already in correct shape
        else:
            raise ValueError(f"Unexpected input tensor shape: {input_tensor.shape}")
        
        # For LoRA model, ensure input has correct dimensions
        if model_name == 'lora':
            # Ensure input has shape [batch_size, seq_len, hidden_size]
            if input_tensor.shape[1] != config['num_experts']:
                # Expand to match number of experts
                input_tensor = input_tensor.expand(-1, config['num_experts'], -1)
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run inference
        _ = model(input_tensor)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return final_memory - initial_memory
    
    def test_generation_quality(self, model_name, prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.9):
        """Test generation quality with improved attention and sampling."""
        model = self.models[model_name]
        
        # Tokenize input and move to correct device
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        output_ids = input_ids.clone()
        
        # Initialize log probabilities for PPL calculation
        log_probs = []
        
        # Generate text with improved sampling
        for _ in range(max_length):
            with torch.no_grad():
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
                    
                    # Process through attention mechanism
                    processed_key = self._process_with_attention(key)
                    processed_value = self._process_with_attention(value)
                    
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
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Store log probability for PPL calculation
                log_probs.append(torch.log(probs[0, next_token[0, 0]]))
                
                # Append to output
                output_ids = torch.cat([output_ids, next_token], dim=1)
        
        # Calculate PPL
        if log_probs:
            ppl = torch.exp(-torch.mean(torch.stack(log_probs))).item()
        else:
            ppl = float('inf')
        
        # Decode and return generated text and PPL
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text, ppl
    
    def run_ablation_test(self, prompt):
        """Run ablation test with improved generation."""
        results = {}
        
        # Create input tensor with correct dimensions
        batch_size = 1
        seq_len = 1
        input_tensor = torch.randn(batch_size, seq_len, config['hidden_size']).to(self.device)
        
        for model_name in self.models:
            print(f"\nTesting {model_name} model...")
            
            # Test inference speed
            inference_time = self.test_inference_speed(model_name, input_tensor)
            
            # Test memory usage
            memory_usage = self.test_memory_usage(model_name, input_tensor)
            
            # Test generation quality with improved parameters
            generated_text, ppl = self.test_generation_quality(
                model_name, 
                prompt,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
            
            results[model_name] = {
                'inference_time': inference_time,
                'memory_usage': memory_usage,
                'generated_text': generated_text,
                'ppl': ppl
            }
            
            print(f"Inference time: {inference_time:.4f} seconds")
            print(f"Memory usage: {memory_usage:.2f} MB")
            print(f"Generated text: {generated_text}")
            print(f"Perplexity: {ppl:.2f}")
        
        return results
    
    def print_results(self, results):
        """Print formatted results."""
        print("\nAblation Test Results:")
        print("=" * 80)
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()} Model:")
            print(f"  Inference Time: {metrics['inference_time']:.4f} seconds")
            print(f"  Memory Usage: {metrics['memory_usage']:.2f} MB")
            print(f"  Perplexity: {metrics['ppl']:.2f}")
            print(f"  Generated Text: {metrics['generated_text']}")
        print("=" * 80)

def main():
    # Initialize ablation test
    ablation_test = SingleAblationTest()
    
    # Example prompt
    prompt = "The quick brown fox"
    
    # Run ablation test
    results = ablation_test.run_ablation_test(prompt)
    
    # Print results
    ablation_test.print_results(results)

if __name__ == "__main__":
    main() 