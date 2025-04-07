import torch
from config import config
import torch.nn  as nn

def generate_data(batch_size, input_size):
    """Generate random data for training."""
    return torch.randn(batch_size, input_size)

def train_model(model, data, target):
    """Training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()
