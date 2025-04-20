import os
from datasets import load_dataset
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
import torch
import random

def generate_data():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Load dataset (using wikitext-2 as an example)
    print("Loading wikitext-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    # Combine all text
    train_text = ""
    test_text = ""
    
    # Process training data
    print("Processing training data...")
    for text in dataset['train']['text']:
        if len(text.strip()) > 0:  # Skip empty lines
            train_text += text.strip() + "\n"
    
    # Process test data
    print("Processing test data...")
    for text in dataset['test']['text']:
        if len(text.strip()) > 0:  # Skip empty lines
            test_text += text.strip() + "\n"
    
    # Save to files
    print("Saving data to files...")
    with open('data/train.txt', 'w', encoding='utf-8') as f:
        f.write(train_text)
    
    with open('data/test.txt', 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    # Print statistics
    train_tokens = len(tokenizer.encode(train_text))
    test_tokens = len(tokenizer.encode(test_text))
    
    print(f"\nData generation complete!")
    print(f"Training data: {train_tokens} tokens")
    print(f"Testing data: {test_tokens} tokens")
    print(f"Files saved to data/train.txt and data/test.txt")

if __name__ == "__main__":
    generate_data() 