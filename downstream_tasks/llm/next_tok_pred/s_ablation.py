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
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get model's hidden size
        self.hidden_size = self.model.config.hidden_size
        
        # Update config with model's hidden size
        config['hidden_size'] = self.hidden_size
        
        # Initialize different MoE models
        self.models = {
            'standard': self._create_standard_moe(),
            'lora': self._create_lora_moe(),
            #'compressed': self._create_compressed_moe(),
            'adaptive': self._create_adaptive_moe(),
            'pikv': self._create_pikv_moe()
        }
        
        # Move models to device
        self.device = config['device']
        for model in self.models.values():
            model.to(self.device)
    
    def _create_standard_moe(self):
        """Create standard MoE model."""
        from core.single.normal_moe import StandardMoE
        return StandardMoE()
    
    def _create_lora_moe(self):
        """Create LoRA MoE model."""
        from core.single.lora_moe import LoRAMoE
        # Use the same number of experts as in config
        return LoRAMoE(rank=config['num_experts'], alpha=1.0)
    
    def _create_compressed_moe(self):
        """Create compressed MoE model."""
        from core.single.kvc_moe import KVCacheMoE
        return KVCacheMoE()
    
    def _create_adaptive_moe(self):
        """Create adaptive routing MoE model."""
        from core.single.routing_moe import RoutingMoE
        return RoutingMoE()
    
    def _create_pikv_moe(self):
        """Create PiKV MoE model."""
        return PiKVMoE(rank=4, alpha=1.0)
    
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
    
    def test_generation_quality(self, model_name, prompt, max_length=50):
        """Test generation quality of a model."""
        model = self.models[model_name]
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Convert input_ids to embeddings (simple projection)
        input_embeddings = torch.nn.functional.one_hot(
            input_ids, 
            num_classes=self.tokenizer.vocab_size
        ).float()  # [batch_size, seq_len, vocab_size]
        
        # Project to hidden size
        projection = torch.nn.Linear(
            self.tokenizer.vocab_size, 
            config['hidden_size']
        ).to(self.device)
        input_embeddings = projection(input_embeddings)  # [batch_size, seq_len, hidden_size]
        
        # For LoRA model, ensure input has correct dimensions
        if model_name == 'lora':
            # Ensure input has shape [batch_size, seq_len, hidden_size]
            if input_embeddings.shape[1] != config['num_experts']:
                # Expand to match number of experts
                input_embeddings = input_embeddings.expand(-1, config['num_experts'], -1)
        
        # Generate text
        output_embeddings = input_embeddings.clone()
        for _ in range(max_length):
            # Get model outputs
            with torch.no_grad():
                # Process through MoE
                outputs = model(output_embeddings)  # [batch_size, seq_len, hidden_size]
                
                # For LoRA model, take mean across experts
                if model_name == 'lora':
                    # Ensure we're taking mean across the correct dimension
                    outputs = outputs.mean(dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
                
                # Get next token logits
                next_token_logits = torch.matmul(
                    outputs[:, -1:, :],  # [batch_size, 1, hidden_size]
                    projection.weight  # [hidden_size, vocab_size]
                )  # [batch_size, 1, vocab_size]
                
                # Get next token ID
                next_token_id = torch.argmax(next_token_logits[:, 0], dim=-1)  # [batch_size]
                
                # Get next token embedding
                next_token_embedding = torch.nn.functional.one_hot(
                    next_token_id, 
                    num_classes=self.tokenizer.vocab_size
                ).float()  # [batch_size, vocab_size]
                next_token_embedding = projection(next_token_embedding)  # [batch_size, hidden_size]
                next_token_embedding = next_token_embedding.unsqueeze(1)  # [batch_size, 1, hidden_size]
                
                # For LoRA model, expand to match number of experts
                if model_name == 'lora':
                    next_token_embedding = next_token_embedding.expand(-1, config['num_experts'], -1)
                
                # Concatenate with previous outputs
                output_embeddings = torch.cat([output_embeddings, next_token_embedding], dim=1)
        
        # Convert embeddings back to token IDs
        output_logits = torch.matmul(
            output_embeddings,  # [batch_size, seq_len, hidden_size]
            projection.weight  # [hidden_size, vocab_size]
        )  # [batch_size, seq_len, vocab_size]
        output_ids = torch.argmax(output_logits, dim=-1)  # [batch_size, seq_len]
        
        # Decode and return generated text
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def run_ablation_test(self, prompt):
        """Run ablation test for all models."""
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
            
            # Test generation quality
            generated_text = self.test_generation_quality(model_name, prompt)
            
            results[model_name] = {
                'inference_time': inference_time,
                'memory_usage': memory_usage,
                'generated_text': generated_text
            }
            
            print(f"Inference time: {inference_time:.4f} seconds")
            print(f"Memory usage: {memory_usage:.2f} MB")
            print(f"Generated text: {generated_text}")
        
        return results
    
    def print_results(self, results):
        """Print formatted results."""
        print("\nAblation Test Results:")
        print("=" * 80)
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()} Model:")
            print(f"  Inference Time: {metrics['inference_time']:.4f} seconds")
            print(f"  Memory Usage: {metrics['memory_usage']:.2f} MB")
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