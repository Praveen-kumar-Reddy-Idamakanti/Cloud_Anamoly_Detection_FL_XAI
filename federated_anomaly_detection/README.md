# Federated Autoencoder for Cloud Anomaly Detection

This project implements a federated learning system for detecting anomalies in cloud activity logs using autoencoders. The system is built with TensorFlow/Keras and Flower for federated learning.

## Project Structure

```
federated_anomaly_detection/
├── data/                   # Directory for node data (auto-created)
├── logs/                   # Training logs and saved models (auto-created)
├── models/
│   └── autoencoder.py      # Autoencoder model definition
├── utils/
│   └── data_utils.py       # Data loading and generation utilities
├── client.py               # Flower client implementation
├── server.py               # Flower server implementation
├── generate_data.py        # Script to generate synthetic data
└── requirements.txt        # Python dependencies
```

## Setup

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Generating Synthetic Data

Generate synthetic cloud activity data for multiple nodes:

```bash
python generate_data.py --num_nodes 3 --samples_per_node 1000 --num_features 20 --output_dir data
```

This will create CSV files (`node_1.csv`, `node_2.csv`, etc.) in the `data` directory.

## Running the Federated Learning System

### 1. Start the Server

In a terminal, run:

```bash
python server.py --input_dim 20 --min_clients 2 --num_rounds 10
```

### 2. Start Clients

Open separate terminal windows for each client and run:

```bash
# Client 1
python client.py --node_id 1 --data_path data/node_1.csv --server_address 0.0.0.0:8080

# Client 2 (in a new terminal)
python client.py --node_id 2 --data_path data/node_2.csv --server_address 0.0.0.0:8080

# Client 3 (in a new terminal)
python client.py --node_id 3 --data_path data/node_3.csv --server_address 0.0.0.0:8080
```

## Monitoring Training

Training logs and TensorBoard events are saved in the `logs/` directory. To monitor training with TensorBoard:

```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006` in your browser.

## How It Works

1. **Data Generation**: Synthetic cloud activity data is generated with some anomalies injected.
2. **Federated Training**:
   - Each client trains a local autoencoder on its own data.
   - The server aggregates model updates from clients using federated averaging.
   - The aggregated model is sent back to clients for the next round of training.
3. **Anomaly Detection**:
   - After training, the autoencoder can detect anomalies by measuring reconstruction error.
   - Data points with high reconstruction error are flagged as anomalies.

## Customization

- **Model Architecture**: Modify the `AnomalyDetector` class in `models/autoencoder.py`.
- **Data Generation**: Adjust parameters in `generate_data.py` or modify `utils/data_utils.py`.
- **Training Parameters**: Change hyperparameters in `server.py` and `client.py`.

## Future Enhancements

- Add support for real cloud activity logs
- Implement differential privacy for enhanced security
- Add explainability (XAI) components to understand model decisions
- Deploy to a real federated learning environment with multiple physical nodes
