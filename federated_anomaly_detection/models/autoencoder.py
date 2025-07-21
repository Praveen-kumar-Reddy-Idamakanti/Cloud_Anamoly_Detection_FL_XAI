import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super(AnomalyDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def train_epoch(self, dataloader, optimizer, device):
        """Train the model for one epoch"""
        self.train()
        total_loss = 0.0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Handle case where batch is (data, target)
                
            batch = batch.to(device)
            
            # Forward pass
            reconstructed = self(batch)
            loss = F.mse_loss(reconstructed, batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.size(0)
            
        return total_loss / len(dataloader.dataset)
    
    def evaluate(self, dataloader, device):
        """Evaluate the model on the given dataset"""
        self.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # Handle case where batch is (data, target)
                    
                batch = batch.to(device)
                reconstructed = self(batch)
                loss = F.mse_loss(reconstructed, batch, reduction='sum')
                total_loss += loss.item()
                
        return total_loss / len(dataloader.dataset)
    
    def detect_anomaly(self, dataloader, device, threshold: float = None):
        """Detect anomalies based on reconstruction error"""
        self.eval()
        all_errors = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # Handle case where batch is (data, target)
                    
                batch = batch.to(device)
                reconstructed = self(batch)
                mse = F.mse_loss(reconstructed, batch, reduction='none').mean(dim=1)
                all_errors.append(mse.cpu().numpy())
        
        errors = np.concatenate(all_errors)
        
        if threshold is None:
            # Use 95th percentile as threshold if not provided
            threshold = np.percentile(errors, 95)
            
        anomalies = errors > threshold
        return anomalies, errors, threshold

def create_model(input_dim: int, learning_rate: float = 0.001, device: str = "cpu") -> Tuple[AnomalyDetector, torch.optim.Optimizer]:
    """
    Create and return the autoencoder model and optimizer
    
    Args:
        input_dim: Number of input features
        learning_rate: Learning rate for the optimizer
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (model, optimizer)
    """
    model = AnomalyDetector(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer
