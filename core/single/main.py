import torch
from config import config
from pikv_moe import PiKVMoE
from normal_moe import StandardMoE
from utils import generate_data, train_model

def compare_models():
    # Generate some random data for training
    data = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    target = generate_data(config['batch_size'], config['hidden_size']).to(config['device'])
    
    # Initialize models
    model_pikv = PiKVMoE().to(config['device'])
    model_standard = StandardMoE().to(config['device'])
    
    # Train both models and compare losses
    for epoch in range(config['epochs']):
        loss_pikv = train_model(model_pikv, data, target)
        loss_standard = train_model(model_standard, data, target)
        print(f"Epoch {epoch+1}/{config['epochs']} - PiKV Loss: {loss_pikv:.4f}, Standard MoE Loss: {loss_standard:.4f}")

if __name__ == "__main__":
    compare_models()
