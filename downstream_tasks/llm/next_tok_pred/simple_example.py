import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from core.single.pikv_moe import PiKVMoE
from core.single.config import config
from tqdm import tqdm

class SimpleDataset(Dataset):
    def __init__(self, sequences, max_length=16):
        self.sequences = sequences
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Ensure sequence length is correct
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        elif len(sequence) < self.max_length:
            sequence = sequence + [0] * (self.max_length - len(sequence))
        
        # Split into input and target
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        
        return torch.tensor(input_ids), torch.tensor(target_ids)

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256):
        super(SimpleModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pikv = PiKVMoE(rank=4, alpha=1.0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize KV cache with correct dimensions
        self.pikv.kv_caches = nn.ModuleList([
            KVCache(size=16, hidden_size=hidden_size)
            for _ in range(config['num_experts'])
        ])
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, hidden_size]
        x = self.pikv(x)  # [batch_size, seq_len, hidden_size]
        x = self.fc(x)  # [batch_size, seq_len, vocab_size]
        return x

class KVCache(nn.Module):
    def __init__(self, size, hidden_size):
        super(KVCache, self).__init__()
        self.size = size
        self.hidden_size = hidden_size
        
        # Initialize tensors with correct dimensions
        self.register_buffer('keys', torch.zeros(size, hidden_size))
        self.register_buffer('values', torch.zeros(size, hidden_size))
        self.register_buffer('importance', torch.zeros(size))
    
    def update(self, idx, key, value, importance):
        # Ensure key and value have correct dimensions
        if len(key.shape) == 3:  # [batch_size, seq_len, hidden_size]
            key = key.mean(dim=0).mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0).mean(dim=0)  # [hidden_size]
        elif len(key.shape) == 2:  # [seq_len, hidden_size]
            key = key.mean(dim=0)  # [hidden_size]
            value = value.mean(dim=0)  # [hidden_size]
        
        # Update cache
        self.keys[idx] = key
        self.values[idx] = value
        self.importance[idx] = importance.mean().item()
    
    def get_all(self):
        return self.values.mean(dim=0)  # [hidden_size]
    
    def set_all(self, data):
        if data is not None:
            self.values.copy_(data.unsqueeze(0).expand(self.size, -1))

def train(model, dataloader, device, num_epochs=5):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for input_ids, target_ids in progress_bar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create simple sequences for training
    sequences = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [3, 4, 5, 6, 7, 8, 9, 10],
        [4, 5, 6, 7, 8, 9, 10, 11],
    ]
    
    # Create dataset and dataloader
    dataset = SimpleDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    vocab_size = 12  # We have tokens from 1 to 11
    model = SimpleModel(vocab_size).to(device)
    
    # Train model
    print("Training model...")
    train(model, dataloader, device)
    
    # Test prediction
    print("\nTesting predictions:")
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7]]).to(device)
        logits = model(test_input)
        predicted = torch.argmax(logits[0, -1, :]).item()
        print(f"Input sequence: {test_input[0].tolist()}")
        print(f"Predicted next token: {predicted}")

if __name__ == "__main__":
    main() 