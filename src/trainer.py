from community_contrastive import *
from cross_modal_contrastive import compute_cross_modal_contrastive_loss
from graph_contrastive import *
from load_data import get_adj_matrices, get_y
from clustering_metrics import calculate_clustering_metrics
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

torch.autograd.set_detect_anomaly(True)

class GraphClusteringTrainer:
    def __init__(self, batch_size, model, optimizer, num_epochs, device, text_data, image_data, graph_file, csv_file, hidden_dim, walks_per_node, walk_length, context_size, aas, theta, std):
        """
        Args:
            model: The model for graph clustering.
            graph: The DGL graph object.
            optimizer: Optimizer for training.
            num_epochs: Number of training epochs.
            device: Device to train on ('cuda' or 'cpu').
        """
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.text_data, self.image_data = text_data, image_data
        self.hidden_dim = hidden_dim
        self.walks_per_node, self.walk_length, self.context_size = walks_per_node, walk_length, context_size
        self.adj_matrix, self.norm_adj_matrix = get_adj_matrices(graph_file)
        self.n = self.norm_adj_matrix.size(0)
        self.y = get_y(csv_file)
        self.num_class = torch.unique(self.y).numel()
        self.kmeans = KMeans(n_clusters=self.y.max().item()+1, n_init=10)
        self.mapping = torch.zeros(self.n, dtype=torch.int64).to(self.device)
        self.aas = aas
        self.theta = theta
        self.std = std

    def train(self):
        """
        Train the model with contrastive learning.
        """
        best_nmi = 0
        best_set = 0, 0, 0, 0, 0
        
        # Load modalities
        text_data = torch.FloatTensor(self.text_data).to(self.device)
        image_data = torch.FloatTensor(self.image_data).to(self.device)

        margin = 0.001

        for _ in range(self.num_epochs):
            self.model.train()
            z1, z2, h = self.model(text_data, image_data, self.norm_adj_matrix)

            loss1, loss2, loss3 = 0, 0, 0

            # Dynamic threshold based on similarity distribution
            threshold = determine_threshold(self.adj_matrix, z1, z2, self.std)
            
            # Loss 1: Cross-modal contrastive loss
            if self.batch_size == -1:
                loss1 = compute_cross_modal_contrastive_loss(z1, z2, h, margin=margin)
            else:
                # Batch processing for large datasets
                z1_cpu, z2_cpu, h_cpu = z1.cpu(), z2.cpu(), h.cpu()
                dataset = FeatureDataset(z1_cpu, z2_cpu, h_cpu)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True)
                total_batches = len(dataloader)

                for z1_batch, z2_batch, h_batch in dataloader:
                    z1_batch = z1_batch.to(self.device).requires_grad_(True)
                    z2_batch = z2_batch.to(self.device).requires_grad_(True) 
                    h_batch = h_batch.to(self.device).requires_grad_(True)
                    batch_loss1 = compute_cross_modal_contrastive_loss(z1_batch, z2_batch, h_batch, margin=margin)
                    loss1 += batch_loss1
                loss1 /= total_batches

            # Loss 2: Graph contrastive loss with attribute-aware filtering
            if not self.aas:
                pos_rw = pos_sample(self.adj_matrix, self.walks_per_node, self.walk_length, self.context_size)
                neg_rw = neg_sample(self.adj_matrix, self.walks_per_node, self.walk_length, self.context_size)
            else:
                filtered_adj_matrix = filter_adjacency_by_similarity(self.adj_matrix, z1, z2, threshold=threshold)
                pos_rw = pos_sample(filtered_adj_matrix, self.walks_per_node, self.walk_length, self.context_size)
                neg_rw = neg_sample(self.adj_matrix, self.walks_per_node, self.walk_length, self.context_size)
            
            unique = torch.unique(torch.cat((pos_rw, neg_rw), dim=-1))
            self.mapping.scatter_(0, unique.to(self.device), torch.arange(unique.size(0)).to(self.device))
            loss2 = get_graph_contrastive_loss(pos_rw.to(self.device), neg_rw.to(self.device), h, self.hidden_dim, self.mapping)

            # Evaluate clustering performance
            with torch.no_grad():
                y_pred = self.kmeans.fit_predict(h.cpu().detach().numpy())
                acc, nmi, f1_macro, adjscore, cs = calculate_clustering_metrics(self.y.cpu().numpy(), y_pred)
                if nmi > best_nmi:
                    best_nmi = nmi
                    best_set = acc, nmi, f1_macro, adjscore, cs

            # Loss 3: Community loss
            if self.theta != -1:
                centroids_tensor = torch.from_numpy(self.kmeans.cluster_centers_).float().to(self.device)
                labels_tensor = torch.from_numpy(y_pred).to(self.device)

                # Standardize and normalize features
                def standardize(tensor, dim=0):
                    mean = torch.mean(tensor, dim=dim, keepdim=True)
                    std = torch.std(tensor, dim=dim, keepdim=True)
                    return (tensor - mean) / std

                h = standardize(h, dim=0)
                h = F.normalize(h, p=2, dim=1)
                centroids_tensor = standardize(centroids_tensor, dim=0)
                centroids_tensor = F.normalize(centroids_tensor, p=2, dim=1)

                loss3 = cluster_loss(h, centroids_tensor, labels_tensor, positive_threshold_percent=self.theta)

            # Compute total loss with normalization
            # print(loss1, loss2, loss3)
            loss = loss1/loss1.detach() + loss2/loss2.detach() + (loss3/loss3.detach() if loss3 != 0 else 0)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        print(f"Best metrics: {best_set}")

class FeatureDataset(Dataset):
    def __init__(self, z1, z2, h):
        """
        Initialize dataset with three features
        Args:
            z1: Tensor containing first feature
            z2: Tensor containing second feature
            h: Tensor containing third feature
        """
        self.z1 = z1
        self.z2 = z2
        self.h = h
        
    def __len__(self):
        """Return total number of samples"""
        return len(self.z1)
    
    def __getitem__(self, idx):
        """Return one sample at position idx"""
        return self.z1[idx], self.z2[idx], self.h[idx]
