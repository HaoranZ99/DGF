import numpy as np
import dgl
import torch
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer


def load_dataset(dataset):
    """Load dataset based on the specified name."""
    if dataset == "Movies":
        text_file = './MAG/Movies/TextFeature/Movies_gemma_7b_256_mean.npy'
        image_file = './MAG/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy'
        graph_file = './MAG/Movies/MoviesGraph.pt'
        csv_file = './MAG/Movies/Movies.csv'
    elif dataset == "Toys":
        text_file = './MAG/Toys/TextFeature/Toys_gemma_7b_256_mean.npy'
        image_file = './MAG/Toys/ImageFeature/Toys_openai_clip-vit-large-patch14.npy'
        graph_file = './MAG/Toys/ToysGraph.pt'
        csv_file = './MAG/Toys/Toys.csv'
    elif dataset == "RedditS":
        text_file = './MAG/RedditS/TextFeature/RedditS_gemma_7b_64_mean.npy'
        image_file = './MAG/RedditS/ImageFeature/RedditS_openai_clip-vit-large-patch14.npy'
        graph_file = './MAG/RedditS/RedditSGraph.pt'
        csv_file = './MAG/RedditS/RedditS.csv'
    elif dataset == "Grocery":
        text_file = './MAG/Grocery/TextFeature/Grocery_gemma_7b_256_mean.npy'
        image_file = './MAG/Grocery/ImageFeature/Grocery_openai_clip-vit-large-patch14.npy'
        graph_file = './MAG/Grocery/GroceryGraph.pt'
        csv_file = './MAG/Grocery/Grocery.csv'
    elif dataset == "GroceryS":
        text_file = './MAG/GroceryS/TextFeature/GroceryS_gemma_7b_256_mean.npy'
        image_file = './MAG/GroceryS/ImageFeature/GroceryS_openai_clip-vit-large-patch14.npy'
        graph_file = './MAG/GroceryS/GrocerySGraph.pt'
        csv_file = './MAG/GroceryS/GroceryS.csv'
    elif dataset == "Photo":
        text_file = './MAG/Photo/TextFeature/Photo_gemma_7b_256_mean.npy'
        image_file = './MAG/Photo/ImageFeature/Photo_openai_clip-vit-large-patch14.npy'
        graph_file = './MAG/Photo/PhotoGraph.pt'
        csv_file = './MAG/Photo/Photo.csv'
    elif dataset == "Arts":
        text_file = './MAG/Arts/TextFeature/Arts_gemma_7b_256_mean.npy'
        image_file = './MAG/Arts/ImageFeature/Arts_openai_clip-vit-large-patch14.npy'
        graph_file = './MAG/Arts/ArtsGraph.pt'
        csv_file = './MAG/Arts/Arts.csv'
    elif dataset == "Reddit":
        text_file = './MAG/Reddit/TextFeature/Reddit_gemma_7b_100_mean.npy'
        image_file = './MAG/Reddit/ImageFeature/Reddit_openai_clip-vit-large-patch14.npy'
        graph_file = './MAG/Reddit/RedditGraph.pt'
        csv_file = './MAG/Reddit/Reddit.csv'
    return text_file, image_file, graph_file, csv_file

def get_dataset_features(text_file, image_file):
    """Load and preprocess text and image features."""
    text_data = np.load(text_file)
    image_data = np.load(image_file)
    
    # Handle infinite values
    text_data = np.where(np.isinf(text_data), np.nan, text_data)
    image_data = np.where(np.isinf(image_data), np.nan, image_data)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    text_data = imputer.fit_transform(text_data)
    image_data = imputer.fit_transform(image_data)

    # L2 normalization
    scaler = Normalizer(norm='l2')
    text_data = scaler.fit_transform(text_data)
    image_data = scaler.fit_transform(image_data)

    return text_data, image_data

def norm_adj(graph):
    """Compute normalized adjacency matrix: D^(-1/2) * A * D^(-1/2)."""
    D = torch.sparse.mm(graph, torch.ones(graph.size(0), 1)).view(-1)
    indices = [[i for i in range(graph.size(0))], [i for i in range(graph.size(0))]]
    D = torch.sparse_coo_tensor(torch.tensor(indices), 1/(D**0.5), graph.size())
    ADinv = torch.sparse.mm(D, torch.sparse.mm(graph, D))
    ADinv = ADinv.coalesce()
    ADinv._values()[torch.isinf(ADinv._values())] = 0
    return ADinv

def get_adj_matrices(graph_file):
    """Load graph and return both original and normalized adjacency matrices."""
    graph_data = dgl.load_graphs(graph_file)[0]
    adj_matrix = graph_data[0].adjacency_matrix(scipy_fmt="csr")
    
    # Make undirected and remove self-loops
    adj_matrix = adj_matrix.maximum(adj_matrix.transpose().tocsr())
    adj_matrix.setdiag(0)
    adj_matrix.eliminate_zeros()
    
    # Convert to sparse tensor
    indices = np.vstack((adj_matrix.nonzero()))
    values = adj_matrix.data
    sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.tensor(indices, dtype=torch.long),
        values=torch.tensor(values, dtype=torch.float),
        size=adj_matrix.shape
    )
    adj_matrix = sparse_tensor.float()
    norm_adj_matrix = norm_adj(adj_matrix)
    
    return adj_matrix, norm_adj_matrix

def get_y(csv_file):
    """Load labels from CSV file."""
    df = pd.read_csv(csv_file)
    return torch.tensor(df['label'].values)
