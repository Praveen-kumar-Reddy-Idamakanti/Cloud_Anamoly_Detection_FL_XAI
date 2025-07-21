import os
import argparse
from utils.data_utils import save_node_data

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic cloud activity data for federated learning")
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=3,
        help="Number of nodes to generate data for"
    )
    parser.add_argument(
        "--samples_per_node",
        type=int,
        default=1000,
        help="Number of samples per node"
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=20,
        help="Number of features in the dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save the generated data"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate data for each node
    for node_id in range(1, args.num_nodes + 1):
        save_node_data(
            node_id=node_id,
            data_dir=args.output_dir,
            n_samples=args.samples_per_node,
            n_features=args.num_features
        )
    
    print(f"Generated data for {args.num_nodes} nodes in '{args.output_dir}' directory")

if __name__ == "__main__":
    main()
