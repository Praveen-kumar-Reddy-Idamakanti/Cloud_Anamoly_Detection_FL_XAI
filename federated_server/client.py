import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from typing import Dict, List, Tuple
import flwr as fl
from collections import OrderedDict
import numpy as np

# Define the same model architecture as in server.py
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_data():
    """Load MNIST training and test data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download training data
    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    
    # Split training set into 10 clients
    num_clients = 10
    client_size = len(trainset) // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = (i + 1) * client_size if i < num_clients - 1 else len(trainset)
        client_datasets.append(torch.utils.data.Subset(trainset, range(start_idx, end_idx)))
    
    # Test set
    testset = datasets.MNIST("./data", train=False, transform=transform)
    
    return client_datasets, testset

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, trainset, testset, device):
        self.cid = cid
        self.device = device
        
        # Split data into train/validation (80/20)
        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            trainset, [train_size, val_size]
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=32, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=32, shuffle=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=32, shuffle=False
        )
        
        # Create model
        self.model = Net().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
    
    # Remove to_client method as we'll use start_numpy_client directly
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Train for one epoch
        self.model.train()
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        
        # Return updated model parameters and results
        return self.get_parameters({}), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        loss = loss / total
        
        return float(loss), len(self.test_loader.dataset), {"accuracy": accuracy}

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load data for this client
    client_datasets, testset = load_data()
    client_id = int(cid)
    
    # Create client
    return FlowerClient(
        cid=cid,
        trainset=client_datasets[client_id],
        testset=testset,
        device=device
    )

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start a Flower client')
    parser.add_argument('--cid', type=str, required=True, help='Client ID (0-9)')
    parser.add_argument('--server-address', type=str, default="127.0.0.1:8080",
                      help='Server address (default: 127.0.0.1:8080)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Starting client {args.cid} on {device}")
    
    try:
        # Load data for this client
        client_datasets, testset = load_data()
        client_id = int(args.cid)
        
        print(f"Client {client_id} loading data...")
        
        # Create client
        client = FlowerClient(
            cid=str(client_id),
            trainset=client_datasets[client_id],
            testset=testset,
            device=device
        )
        
        print(f"Client {client_id} connecting to server at {args.server_address}...")
        
        # Start client
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client
        )
        
        print(f"Client {client_id} finished successfully!")
        
    except Exception as e:
        print(f"Error in client {args.cid}: {str(e)}")
        import traceback
        traceback.print_exc()
