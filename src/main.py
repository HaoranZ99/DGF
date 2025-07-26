import argparse
import torch
import torch.optim as optim
from model import GraphClusteringModel
from trainer import GraphClusteringTrainer
from load_data import get_dataset_features, get_y, load_dataset

def main():
    parser = argparse.ArgumentParser(description="Graph Clustering Trainer")
    parser.add_argument('--dataset', type=str, default='Movies')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--walk_length', type=int, default=5)
    parser.add_argument('--context_size', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--aas', action="store_true")
    parser.add_argument('--theta', type=float, default=0.3)
    parser.add_argument('--std', type=float, default=1)
    args = parser.parse_args()

    # Load dataset
    text_file, image_file, graph_file, csv_file = load_dataset(args.dataset)
    text_data, image_data = get_dataset_features(text_file, image_file)
    y = get_y(csv_file)

    # Initialize model and optimizer
    device = torch.device(args.device)
    model = GraphClusteringModel(
        d1=text_data.shape[1], 
        d2=image_data.shape[1], 
        hidden_dim=args.hidden_dim, 
        alpha=args.alpha, 
        beta=args.beta, 
        num_layers=args.num_layers, 
        device=device,
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # Create and run trainer
    trainer = GraphClusteringTrainer(
        batch_size=args.batch_size,
        model=model,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        text_data=text_data,
        image_data=image_data,
        graph_file=graph_file,
        csv_file=csv_file,
        hidden_dim=args.hidden_dim,
        walks_per_node=args.walks_per_node,
        walk_length=args.walk_length,
        context_size=args.context_size,
        aas=args.aas,
        theta=args.theta,
        std=args.std,
    )

    trainer.train()

if __name__ == "__main__":
    main()
