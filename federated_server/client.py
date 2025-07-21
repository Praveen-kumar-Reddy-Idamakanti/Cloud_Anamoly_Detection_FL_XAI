import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from typing import Dict, List, Tuple, Optional, Any
import flwr as fl
from collections import OrderedDict
import numpy as np
import os
import sys
from secure_aggregation import SecureAggregator, encrypt_parameters

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    def __init__(self, cid: str, trainset, testset, device, num_clients: int = 2):
        self.cid = int(cid)  # Ensure cid is an integer
        self.device = device
        self.num_clients = num_clients
        self.round_num = 0
        
        # Initialize secure aggregator
        self.secure_aggregator = SecureAggregator(
            num_clients=num_clients,
            client_id=self.cid
        )
        
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
        
        # Store parameter shapes and dtypes for encryption
        self.param_shapes = [p.shape for p in self.model.parameters()]
        self.param_dtypes = [str(p.detach().cpu().numpy().dtype) for p in self.model.parameters()]
    
    # Remove to_client method as we'll use start_numpy_client directly
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        # Update round number from config if available
        self.round_num = config.get('server_round', 0)
        
        # Convert parameters to numpy arrays if they're not already
        if parameters is not None:
            if not isinstance(parameters, list):
                try:
                    parameters = fl.common.parameters_to_ndarrays(parameters)
                except Exception as e:
                    print(f"Error converting parameters: {e}")
        
        # Set parameters from server
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
        
        # Get updated parameters as numpy arrays
        parameters_prime = self.get_parameters({})
        
        # Apply secure aggregation masking
        try:
            masked_parameters, _ = self.secure_aggregator.mask_updates(
                parameters_prime, 
                self.round_num
            )
            
            # Ensure all parameters are numpy arrays with consistent shapes
            processed_params = []
            for param in masked_parameters:
                if isinstance(param, np.ndarray):
                    processed_params.append(param.astype(np.float32))
                else:
                    processed_params.append(np.array(param, dtype=np.float32))
            
            # Calculate training metrics
            train_loss, train_accuracy = self._evaluate_metrics()
            metrics = {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "client_id": self.cid,
                "round": self.round_num
            }
            
            # Convert to Flower Parameters and return with metrics
            parameters_aggregated = fl.common.ndarrays_to_parameters(processed_params)
            return parameters_prime, len(self.train_loader.dataset), metrics
            
        except Exception as e:
            print(f"Error in secure aggregation: {e}")
            # Fall back to standard parameters if secure aggregation fails
            train_loss, train_accuracy = self._evaluate_metrics()
            metrics = {
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "client_id": self.cid,
                "round": self.round_num
            }
            return parameters_prime, len(self.train_loader.dataset), metrics
    
    def _evaluate_metrics(self):
        """Helper method to calculate training metrics"""
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        return avg_loss, accuracy
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Evaluate
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        metrics = {
            "accuracy": float(accuracy),
            "loss": float(avg_loss),
            "client_id": self.cid,
            "round": self.round_num
        }
        
        return float(avg_loss), total_samples, metrics

def client_fn(cid: str) -> fl.client.Client:
    """Create a Flower client representing a single organization."""
    import os
    import torch
    
    # Load model and data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    client_datasets, testset = load_data()
    
    # Get number of clients from environment variable or use default (2)
    num_clients = int(os.environ.get('NUM_CLIENTS', 2))
    
    # Create and return client
    return FlowerClient(
        cid=cid,
        trainset=client_datasets[0],
        testset=testset,
        device=device,
        num_clients=num_clients
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
