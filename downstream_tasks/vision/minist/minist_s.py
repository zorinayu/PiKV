import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from core.single.module.pikv_scheduling import CacheSchedulingManager, SchedulingPolicy

# Modify the configuration for image data (MNIST)
config = {
    'input_size': 28 * 28,  # Flattened size for MNIST images (28x28 pixels)
    'batch_size': 64,  # Set batch size for training
    'num_experts': 4,  # Number of experts in MoE
    'hidden_size': 256,  # Hidden size for each expert
    'num_heads': 8,  # Number of attention heads
    'kv_cache_size': 128,  # Cache size for each expert
    'epochs': 10,  # Number of epochs for training
    'learning_rate': 1e-3,  # Learning rate
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'pyramidal_cache': True,  # Use pyramidal cache allocation strategy
    'cache_decrement': 10,  # Cache size decrement for each layer
    'num_layers': 5,  # Number of layers in the model
}

# Define Convolutional PiKV MoE Model for image data
class PiKVMoE(nn.Module):
    def __init__(self):
        super(PiKVMoE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # After conv1 -> 28x28x32
        # After pool -> 14x14x32
        # After conv2 -> 14x14x64
        # After pool -> 7x7x64
        self.fc_in_features = 64 * 7 * 7  # 64 channels * 7x7 spatial size after pooling
        
        self.fc = nn.Linear(self.fc_in_features, config['hidden_size'])
        self.experts = nn.ModuleList([nn.Linear(config['hidden_size'], config['hidden_size']) for _ in range(config['num_experts'])])
        self.gate = nn.Linear(config['hidden_size'], config['num_experts'])
        self.output_layer = nn.Linear(config['hidden_size'], 10)  # Add output layer for 10 MNIST classes

        # Cache size allocation for each layer
        self.cache_sizes = self.pyramidal_cache_allocation()

        # Initialize PiKV cache manager
        self.cache_manager = CacheSchedulingManager(
            cache_size=config['kv_cache_size'],
            hidden_size=config['hidden_size'],
            policy=SchedulingPolicy.LRU  # Choose a policy
        )

    def pyramidal_cache_allocation(self):
        """
        Calculate the cache size for each layer using the pyramidal allocation policy.
        """
        C1 = config['kv_cache_size']
        d = config['cache_decrement']
        return [C1 - (i - 1) * d for i in range(1, config['num_layers'] + 1)]

    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool

        x = x.view(-1, self.fc_in_features)  # Flatten using the correct dimension
        x = F.relu(self.fc(x))
        
        # Gate scores to select experts
        gate_scores = self.gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)

        expert_output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_output += expert(x) * gate_probs[:, i].unsqueeze(-1)
            
        output = self.output_layer(expert_output)  # Final classification layer

        # Update cache with the current hidden state
        self.cache_manager.update_cache(x.unsqueeze(0), x.unsqueeze(0))

        return output

# Define Convolutional Standard MoE Model for image data
class StandardMoE(nn.Module):
    def __init__(self):
        super(StandardMoE, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc_in_features = 64 * 7 * 7  # 64 channels * 7x7 spatial size after pooling
        
        self.fc = nn.Linear(self.fc_in_features, config['hidden_size'])
        self.experts = nn.ModuleList([nn.Linear(config['hidden_size'], config['hidden_size']) for _ in range(config['num_experts'])])
        self.output_layer = nn.Linear(config['hidden_size'], 10)  # Add output layer for 10 MNIST classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool

        x = x.view(-1, self.fc_in_features)  # Flatten using the correct dimension
        x = F.relu(self.fc(x))
        
        # Directly pass input to all experts
        expert_output = torch.zeros_like(x)
        for expert in self.experts:
            expert_output += expert(x)
            
        output = self.output_layer(expert_output)  # Final classification layer
        return output

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalizing
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# Define the training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    accuracy = 100 * correct / total
    return running_loss / len(train_loader), accuracy

# Define the evaluation function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return running_loss / len(test_loader), accuracy

# Initialize models, criterion, and optimizers
device = config['device']
model_pikv = PiKVMoE().to(device)
model_standard = StandardMoE().to(device)

criterion = nn.CrossEntropyLoss()
optimizer_pikv = optim.Adam(model_pikv.parameters(), lr=config['learning_rate'])
optimizer_standard = optim.Adam(model_standard.parameters(), lr=config['learning_rate'])

# Training and evaluation loop
num_epochs = config['epochs']
for epoch in range(num_epochs):
    # Train both models
    loss_pikv, acc_pikv = train_model(model_pikv, train_loader, criterion, optimizer_pikv, device)
    loss_standard, acc_standard = train_model(model_standard, train_loader, criterion, optimizer_standard, device)
    
    # Evaluate both models
    loss_pikv_test, acc_pikv_test = evaluate_model(model_pikv, test_loader, criterion, device)
    loss_standard_test, acc_standard_test = evaluate_model(model_standard, test_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"PiKV Model - Train Loss: {loss_pikv:.4f}, Train Accuracy: {acc_pikv:.2f}% | Test Loss: {loss_pikv_test:.4f}, Test Accuracy: {acc_pikv_test:.2f}%")
    print(f"Standard MoE Model - Train Loss: {loss_standard:.4f}, Train Accuracy: {acc_standard:.2f}% | Test Loss: {loss_standard_test:.4f}, Test Accuracy: {acc_standard_test:.2f}%")
