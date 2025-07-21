import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
import flwr as fl
from datetime import datetime
import json
import time

from models.autoencoder import create_model
from utils.data_utils import load_node_data, generate_cloud_activity_data

def get_device() -> torch.device:
    """Get the device to run the model on"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AnomalyDetectionClient(fl.client.NumPyClient):
    def __init__(self, node_id: int, data_path: str):
        self.node_id = node_id
        self.data_path = data_path
        self.device = get_device()
        
        # Load data
        try:
            x, _ = load_node_data(node_id, os.path.dirname(data_path))
        except FileNotFoundError:
            print(f"No data file found at {data_path}. Generating synthetic data...")
            x, _ = generate_cloud_activity_data(n_samples=1000, n_features=20, random_state=42+node_id)
        
        # Convert to PyTorch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        # Split data into train/validation (80/20)
        split_idx = int(0.8 * len(x_tensor))
        self.train_data = x_tensor[:split_idx]
        self.val_data = x_tensor[split_idx:]
        
        # Create data loaders
        self.batch_size = 32
        self.train_loader = DataLoader(
            TensorDataset(self.train_data, self.train_data),  # Autoencoder uses input as target
            batch_size=self.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(self.val_data, self.val_data),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Create model and optimizer
        self.input_dim = x.shape[1]
        self.model, self.optimizer = create_model(
            input_dim=self.input_dim,
            learning_rate=0.001,
            device=self.device
        )
        
        # Create log directory
        self.log_dir = f"logs/client_{node_id}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"Client {node_id} initialized with {len(self.train_data)} training samples on {self.device}")
    
    def get_parameters(self, config: Dict[str, str] = None) -> List[np.ndarray]:
        """Get model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Train the model on the local data."""
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Get training config
        epochs = int(config.get("epochs", 1))
        
        # Train the model
        train_losses = []
        for epoch in range(epochs):
            loss = self.model.train_epoch(self.train_loader, self.optimizer, self.device)
            train_losses.append(loss)
            
            # Log metrics
            if epoch % 5 == 0 or epoch == epochs - 1:
                val_loss = self.model.evaluate(self.val_loader, self.device)
                print(f"Client {self.node_id} - Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Calculate metrics
        avg_train_loss = sum(train_losses) / len(train_losses)
        val_loss = self.model.evaluate(self.val_loader, self.device)
        
        # Save model checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'epoch': epochs
        }
        torch.save(checkpoint, os.path.join(self.log_dir, f'model_checkpoint_round_{config.get("server_round", 0)}.pth'))
        
        # Return updated parameters and metrics
        return (
            self.get_parameters(),
            len(self.train_loader.dataset),
            {"train_loss": float(avg_train_loss), "val_loss": float(val_loss)},
        )
    
    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the model on the local test set."""
        self.set_parameters(parameters)
        
        # Evaluate on validation set
        val_loss = self.model.evaluate(self.val_loader, self.device)
        
        # Detect anomalies
        anomalies, errors, threshold = self.model.detect_anomaly(
            self.val_loader, 
            self.device,
            threshold=None  # Auto-determine threshold
        )
        anomaly_ratio = np.mean(anomalies)
        
        # Log results
        results = {
            'val_loss': float(val_loss),
            'anomaly_ratio': float(anomaly_ratio),
            'threshold': float(threshold),
            'num_samples': len(self.val_loader.dataset)
        }
        
        # Save evaluation results
        with open(os.path.join(self.log_dir, f'eval_results_round_{config.get("server_round", 0)}.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return (
            float(val_loss),
            len(self.val_loader.dataset),
            results,
        )

def main():
    parser = argparse.ArgumentParser(description="Federated Anomaly Detection Client")
    parser.add_argument(
        "--node_id", type=int, required=True, help="Unique ID for this client node"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/node_{node_id}.csv",
        help="Path to the node's data file (supports {node_id} formatting)",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address (default: 0.0.0.0:8080)",
    )
    
    args = parser.parse_args()
    
    # Format data path with node_id if needed
    if "{node_id}" in args.data_path:
        args.data_path = args.data_path.format(node_id=args.node_id)
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
    
    # Start Flower client
    client = AnomalyDetectionClient(args.node_id, args.data_path)
    
    print(f"Starting client {args.node_id} connecting to {args.server_address}")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )

if __name__ == "__main__":
    main()
