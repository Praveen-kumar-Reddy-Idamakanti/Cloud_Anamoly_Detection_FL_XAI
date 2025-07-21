import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional, OrderedDict, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
import sys
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
    
    # Define strategy with more lenient client requirements
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,  # Wait for at least 2 clients to be available
        min_fit_clients=2,  # Require at least 2 clients for training
        min_evaluate_clients=2,  # Require at least 2 clients for evaluation
        evaluate_fn=get_eval_fn(net, device),
        on_fit_config_fn=lambda rnd: {"round": rnd, "local_epochs": 1},
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