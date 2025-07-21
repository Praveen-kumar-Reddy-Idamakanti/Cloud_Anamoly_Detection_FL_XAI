import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generate_cloud_activity_data(n_samples=1000, n_features=20, anomaly_ratio=0.05, random_state=42):
    """
    Generate synthetic cloud activity data with some anomalies.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features in the dataset
        anomaly_ratio: Ratio of anomalies in the data
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (data, labels) where labels are 0 for normal and 1 for anomaly
    """
    np.random.seed(random_state)
    
    # Generate normal data (multivariate normal distribution)
    normal_data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Create anomalies by adding outliers to some samples
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Make anomalies by shifting some features significantly
    for idx in anomaly_indices:
        # Select random features to modify
        n_affected = np.random.randint(1, n_features//2 + 1)
        affected_features = np.random.choice(n_features, n_affected, replace=False)
        
        # Add significant deviation to selected features
        normal_data[idx, affected_features] += np.random.uniform(3, 10, n_affected)
    
    # Create labels (0: normal, 1: anomaly)
    labels = np.zeros(n_samples, dtype=int)
    labels[anomaly_indices] = 1
    
    # Scale data to [0, 1]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(normal_data)
    
    return scaled_data, labels

def save_node_data(node_id, data_dir='data', n_samples=1000, n_features=20):
    """
    Generate and save synthetic data for a specific node
    
    Args:
        node_id: ID of the node/client
        data_dir: Directory to save the data
        n_samples: Number of samples to generate
        n_features: Number of features in the dataset
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data with different random seeds for different nodes
    data, labels = generate_cloud_activity_data(
        n_samples=n_samples,
        n_features=n_features,
        random_state=42 + node_id  # Different seed for each node
    )
    
    # Create a DataFrame and save to CSV
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_columns)
    df['is_anomaly'] = labels
    
    # Save to CSV
    file_path = os.path.join(data_dir, f'node_{node_id}.csv')
    df.to_csv(file_path, index=False)
    print(f'Saved data for node {node_id} to {file_path}')

def load_node_data(node_id, data_dir='data'):
    """
    Load data for a specific node
    
    Args:
        node_id: ID of the node/client
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    file_path = os.path.join(data_dir, f'node_{node_id}.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file found for node {node_id} at {file_path}")
    
    df = pd.read_csv(file_path)
    features = df.drop('is_anomaly', axis=1).values
    labels = df['is_anomaly'].values
    
    return features, labels
