import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from llm_models import PiKVLLM, StandardMoELLM
from llm_config import llm_config as config
import wandb
import os
from tqdm import tqdm
import math

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and tokenize text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize text
        self.tokens = tokenizer.encode(text)
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - max_length, max_length // 2):
            sequence = self.tokens[i:i + max_length]
            if len(sequence) == max_length:
                self.sequences.append(sequence)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        return torch.tensor(input_ids), torch.tensor(target_ids)

def get_lr(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

def train_model(model_type='pikv'):
    # Initialize wandb
    wandb.init(project="pikv-llm", name=f"{model_type}-training")
    
    # Set device
    device = config['device']
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create dataset and dataloader
    dataset = TextDataset('data/train.txt', tokenizer, config['max_seq_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize model
    if model_type == 'pikv':
        model = PiKVLLM().to(device)
    else:
        model = StandardMoELLM().to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    total_steps = len(dataloader) * config['epochs']
    warmup_steps = config['warmup_steps']
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        for step, (input_ids, target_ids) in enumerate(progress_bar):
            # Move data to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            loss = criterion(logits.view(-1, config['vocab_size']), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            # Update weights
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate
            lr = get_lr(step + epoch * len(dataloader), warmup_steps, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (step + 1)})
            
            # Log metrics
            if step % 100 == 0:
                wandb.log({
                    'loss': loss.item(),
                    'learning_rate': lr,
                    'epoch': epoch,
                    'step': step + epoch * len(dataloader)
                })
            
            # Save checkpoint
            if (step + epoch * len(dataloader)) % config['save_steps'] == 0:
                save_path = f'checkpoints/{model_type}_epoch{epoch}_step{step}.pt'
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, save_path)
    
    # Save final model
    save_path = f'checkpoints/{model_type}_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

def evaluate_model(model_type='pikv'):
    # Set device
    device = config['device']
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create dataset and dataloader
    dataset = TextDataset('data/test.txt', tokenizer, config['max_seq_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Load model
    if model_type == 'pikv':
        model = PiKVLLM().to(device)
    else:
        model = StandardMoELLM().to(device)
    
    checkpoint = torch.load(f'checkpoints/{model_type}_final.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize metrics
    total_loss = 0
    total_tokens = 0
    correct_tokens = 0
    
    # Evaluation loop
    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc='Evaluating'):
            # Move data to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            loss = nn.CrossEntropyLoss()(logits.view(-1, config['vocab_size']), target_ids.view(-1))
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == target_ids).sum().item()
            
            # Update metrics
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += target_ids.numel()
            correct_tokens += correct
    
    # Calculate final metrics
    avg_loss = total_loss / len(dataset)
    accuracy = correct_tokens / total_tokens
    perplexity = math.exp(avg_loss)
    
    # Log metrics
    wandb.log({
        'eval_loss': avg_loss,
        'eval_accuracy': accuracy,
        'eval_perplexity': perplexity
    })
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }

if __name__ == '__main__':
    # Train both models
    print("Training PiKV model...")
    train_model('pikv')
    print("Training Standard MoE model...")
    train_model('standard')
    
    # Evaluate both models
    print("Evaluating PiKV model...")
    pikv_metrics = evaluate_model('pikv')
    print("Evaluating Standard MoE model...")
    standard_metrics = evaluate_model('standard')
    
    # Print comparison
    print("\nModel Comparison:")
    print(f"{'Metric':<15} {'PiKV':<15} {'Standard MoE':<15}")
    print("-" * 45)
    for metric in ['loss', 'accuracy', 'perplexity']:
        print(f"{metric:<15} {pikv_metrics[metric]:<15.4f} {standard_metrics[metric]:<15.4f}") 