import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from core.single.pikv_moe import PiKVMoE
from core.single.config import config
from tqdm import tqdm

class SimpleTextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize text
        tokens = tokenizer.encode(text)
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(tokens) - max_length, max_length):
            sequence = tokens[i:i + max_length]
            if len(sequence) == max_length:
                input_sequence = sequence[:-1]  # All tokens except last
                target_sequence = sequence[1:]  # All tokens except first
                self.sequences.append((input_sequence, target_sequence))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_sequence, target_sequence = self.sequences[idx]
        return torch.tensor(input_sequence), torch.tensor(target_sequence)

def train_model(model, dataloader, device, num_epochs=5):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for input_ids, target_ids in progress_bar:
            # Move data to device
            input_ids = input_ids.to(device)  # [batch_size, seq_len-1]
            target_ids = target_ids.to(device)  # [batch_size, seq_len-1]
            
            # Forward pass
            logits = model(input_ids)  # [batch_size, seq_len-1, vocab_size]
            
            # Reshape for loss calculation
            # logits: [batch_size * seq_len-1, vocab_size]
            # target_ids: [batch_size * seq_len-1]
            loss = criterion(
                logits.view(-1, logits.size(-1)),  # Flatten all dimensions except vocab_size
                target_ids.view(-1)  # Flatten to 1D
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

def predict_next_token(model, tokenizer, text, device, max_length=32):
    model.eval()
    with torch.no_grad():
        # Tokenize input text
        tokens = tokenizer.encode(text)
        
        # Take the last max_length tokens
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]
        
        # Convert to tensor and add batch dimension
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)  # [1, seq_len]
        
        # Get model prediction
        logits = model(input_ids)  # [1, seq_len, vocab_size]
        
        # Get the last token's prediction
        next_token_logits = logits[0, -1, :]  # [vocab_size]
        next_token_id = torch.argmax(next_token_logits).item()
        
        # Decode the predicted token
        next_token = tokenizer.decode([next_token_id])
        
        return next_token

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Sample text for training
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    A journey of a thousand miles begins with a single step.
    To be or not to be, that is the question.
    All that glitters is not gold.
    """
    
    # Create dataset and dataloader
    dataset = SimpleTextDataset(sample_text, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    model = PiKVMoE(rank=4, alpha=1.0).to(device)
    
    # Train model
    print("Training model...")
    train_model(model, dataloader, device)
    
    # Test prediction
    print("\nTesting predictions:")
    test_texts = [
        "The quick brown",
        "A journey of",
        "To be or",
        "All that"
    ]
    
    for text in test_texts:
        next_token = predict_next_token(model, tokenizer, text, device)
        print(f"Input: '{text}'")
        print(f"Predicted next token: '{next_token}'")
        print()

if __name__ == "__main__":
    main() 