import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional, OrderedDict, Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
import sys
import os
from collections import OrderedDict

# Import secure aggregation utilities
from secure_aggregation import SecureAggregator, decrypt_parameters

# Import client function and model
from client import client_fn, Net

# Command line arguments will be parsed in main()

# Define the model architecture
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

# Custom strategy for secure aggregation
class SecureFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, num_clients: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_clients = num_clients
        self.round_num = 0
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate fit results using secure aggregation."""
        self.round_num = server_round
        
        if not results:
            return None, {}
            
        # Get the first result's parameters
        first_result = results[0][1].parameters
        
        # Convert Parameters to list of numpy arrays
        try:
            # Try to convert to numpy arrays first
            first_params = fl.common.parameters_to_ndarrays(first_result)
            use_secure_aggregation = True
        except Exception as e:
            print(f"Not using secure aggregation: {e}")
            use_secure_aggregation = False
        
        if use_secure_aggregation:
            print(f"[Round {server_round}] Using secure aggregation with {len(results)} clients")
            
            # Collect all masked updates
            masked_updates = []
            num_samples = []
            
            for client_proxy, fit_res in results:
                # Convert Parameters to numpy arrays
                client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                
                # Ensure parameters are numpy arrays with float32 dtype
                client_params = [
                    p.astype(np.float32) if isinstance(p, np.ndarray) 
                    else np.array(p, dtype=np.float32) 
                    for p in client_params
                ]
                
                masked_updates.append(client_params)
                num_samples.append(fit_res.num_examples)
            
            # Average the masked updates
            if masked_updates:
                # Simple average of the masked updates
                # In a real implementation, you would verify and combine the masks
                try:
                    avg_parameters = [
                        np.mean([updates[i] for updates in masked_updates], axis=0)
                        for i in range(len(masked_updates[0]))
                    ]
                    
                    # Convert back to parameters
                    parameters_aggregated = fl.common.ndarrays_to_parameters(avg_parameters)
                except Exception as e:
                    print(f"Error in secure aggregation: {e}")
                    # Fall back to standard FedAvg if there's an error
                    return super().aggregate_fit(server_round, results, failures)
                return parameters_aggregated, {}
        
        # Fall back to standard FedAvg if not using secure aggregation
        return super().aggregate_fit(server_round, results, failures)

# Define Flower strategy
def get_eval_fn(model, device):
    """Return an evaluation function for server-side evaluation."""
    # Load MNIST test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    testset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Set model parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        loss = test_loss / len(testloader)
        print(f"Server-side evaluation accuracy: {accuracy*100:.2f}%")
        return loss, {"accuracy": accuracy}
    
    return evaluate

def main():
    print("Starting Federated Learning Server...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients to expect')
    parser.add_argument('--num_rounds', type=int, default=10, help='Number of federated learning rounds')
    parser.add_argument('--fraction_fit', type=float, default=1.0, help='Fraction of clients used during training')
    parser.add_argument('--fraction_evaluate', type=float, default=1.0, help='Fraction of clients used during validation')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    net = Net().to(device)
    
    # Define secure aggregation strategy
    strategy = SecureFedAvg(
        num_clients=args.num_clients,
        min_available_clients=2,  # Wait for at least 2 clients to be available
        min_fit_clients=2,  # Require at least 2 clients for training
        min_evaluate_clients=2,  # Require at least 2 clients for evaluation
        evaluate_fn=get_eval_fn(net, device),
        on_fit_config_fn=lambda rnd: {
            "server_round": rnd,
            "local_epochs": 1,
            "use_encryption": True  # Enable secure aggregation
        },
    )
    
    print(f"\nFederated Learning Configuration:")
    print(f"- Minimum clients required: {strategy.min_fit_clients}")
    print(f"- Total clients expected: {args.num_clients}")
    print(f"- Number of rounds: {args.num_rounds}")
    
    # Start Flower server with strategy
    print(f"\nStarting server on port {args.port}...")
    print(f"Waiting for {args.num_clients} clients to connect...")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        fl.server.start_server(
            server_address=f"0.0.0.0:{args.port}",
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
            strategy=strategy,
            grpc_max_message_length=1024*1024*1024,  # 1GB
        )
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError in server: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()