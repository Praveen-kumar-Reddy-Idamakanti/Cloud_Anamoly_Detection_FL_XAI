# Federated Learning Server

A basic federated learning server using Flower and TensorFlow.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   python server.py --port 8080
   ```

## Connecting Clients

Clients can connect to this server using the Flower client API. The server will wait for at least 2 clients to connect before starting the federated learning process.

## Server Configuration

- **Port**: 8080 (configurable via `--port` argument)
- **Minimum clients required**: 2
- **Training rounds**: 10
- **Model**: Simple CNN for MNIST classification
