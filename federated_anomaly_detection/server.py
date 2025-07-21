import argparse
from typing import Dict, List, Optional, Tuple, Union, Any
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import numpy as np
import torch
from datetime import datetime
import os
import json

from models.autoencoder import create_model, AnomalyDetector

def get_device() -> torch.device:
    """Get the device to run the model on"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define strategy for federated learning
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        input_dim: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn = None,
        on_fit_config_fn = None,
        initial_parameters = None,
    ) -> None:
        # Store custom parameters
        self.input_dim = input_dim
        self.device = get_device()
        self.best_loss = float('inf')
        self.log_dir = f"logs/server/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.round_num = 0
        
        # Initialize parent class with compatible parameters
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn,
            initial_parameters=initial_parameters,
        )
        
        # Store eval_fn separately since it's not a direct parameter in newer Flower versions
        self.eval_fn = eval_fn

    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store the model"""
        # Call parent's aggregate_fit to handle the weighted averaging
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Save aggregated model
            model = AnomalyDetector(self.input_dim).to(self.device)
            
            # Convert parameters to PyTorch state dict format
            params_dict = zip(model.state_dict().keys(), parameters_to_ndarrays(aggregated_parameters))
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            
            # Save the model
            model_path = os.path.join(self.log_dir, f"model_round_{server_round}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'round': server_round,
                'metrics': aggregated_metrics
            }, model_path)
            print(f"Model saved to {model_path}")
            
            # Log metrics
            metrics_path = os.path.join(self.log_dir, f"round_{server_round}_metrics.json")
            with open(metrics_path, 'w') as f:
                round_metrics = {
                    'server_round': server_round,
                    'aggregated_metrics': aggregated_metrics,
                    'client_metrics': [
                        {
                            'client_id': str(fit_res.metrics.get('client_id', i)),
                            'metrics': {
                                k: float(v) for k, v in fit_res.metrics.items()
                                if k != 'client_id'
                            },
                            'num_examples': fit_res.num_examples
                        }
                        for i, (_, fit_res) in enumerate(results)
                        if fit_res.metrics
                    ]
                }
                json.dump(round_metrics, f, indent=2)
        
        return aggregated_parameters, aggregated_metrics

    def evaluate(
        self, server_round: int, parameters: fl.common.NDArrays
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None

        # Evaluate model
        result = self.eval_fn(server_round, parameters, {})
        if result is None:
            return None
        return result

def get_eval_fn(input_dim: int):
    """Return an evaluation function for server-side evaluation."""
    device = get_device()
    
    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """This function would evaluate the model on a server-side validation set.
        Since we don't have server-side data in this example, we'll return default values.
        In a production environment, you would load a validation set here
        and evaluate the model's performance."""
        # Return default values for loss and metrics
        return 0.0, {"accuracy": 0.0}
    
    return evaluate

def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 5,  # Number of local epochs
        "batch_size": 32,
        "server_round": server_round,  # The current round of federated learning
    }
    return config

def main():
    parser = argparse.ArgumentParser(description="Federated Anomaly Detection Server")
    parser.add_argument(
        "--input_dim",
        type=int,
        default=20,
        help="Number of features in the input data",
    )
    parser.add_argument(
        "--min_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for training",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address (default: 0.0.0.0:8080)",
    )
    
    args = parser.parse_args()
    
    # Create initial model
    device = get_device()
    model, _ = create_model(
        input_dim=args.input_dim,
        learning_rate=0.001,
        device=device
    )
    
    # Get initial parameters
    init_params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(init_params)
    
    # Define strategy
    strategy = SaveModelStrategy(
        input_dim=args.input_dim,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients // 2,
        min_available_clients=args.min_clients,
        eval_fn=get_eval_fn(args.input_dim),
        on_fit_config_fn=fit_config,
        initial_parameters=parameters,
    )
    
    # Start server
    print(f"Starting server on {args.server_address}")
    print(f"Using device: {device}")
    print(f"Input dimension: {args.input_dim}")
    print(f"Minimum clients: {args.min_clients}")
    print(f"Number of rounds: {args.num_rounds}")
    
    # Start Flower server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
