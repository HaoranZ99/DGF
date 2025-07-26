
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_cluster
random_walk = torch.ops.torch_cluster.random_walk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def filter_adjacency_by_similarity(adj, z1, z2, threshold=0.9):
    """Filter edges based on cosine similarity between node embeddings."""
    edge_indices = adj.coalesce().indices()
    rows, cols = edge_indices[0], edge_indices[1]
    
    # Compute cosine similarities for existing edges
    z1_edges = z1[rows]
    z2_edges = z2[cols]
    similarities = torch.sum(z1_edges * z2_edges, dim=1)
    
    # Filter edges above threshold
    mask = similarities >= threshold
    new_indices = edge_indices[:, mask]
    new_values = torch.ones(new_indices.size(1), device=adj.device)
    
    new_adj = torch.sparse_coo_tensor(
        indices=new_indices,
        values=new_values,
        size=adj.size(),
        device=adj.device
    ).coalesce()
    
    return new_adj

def determine_threshold(adj, z1, z2, num_std=0):
    """Determine similarity threshold based on random edge statistics."""
    n = adj.size(0)
    m = adj._nnz()

    # Sample random edges
    rows = torch.randint(0, n, (m,))
    cols = torch.randint(0, n, (m,))

    # Compute similarity statistics
    sim_samples = (z1[rows] * z2[cols]).sum(dim=1)
    mu = sim_samples.mean()
    sigma = sim_samples.std()
    
    return (mu + num_std * sigma).item()

def pos_sample(adj, walks_per_node, walk_length, context_size):
    """Generate positive samples using random walks."""
    nodes = torch.arange(adj.shape[0]).repeat(walks_per_node)
    adj = adj.to_sparse_csr()
    rowptr = adj.crow_indices()
    col = adj.col_indices()
    
    rw = random_walk(rowptr, col, nodes, walk_length, 1.0, 1.0)
    if not isinstance(rw, torch.Tensor):
        rw = rw[0]

    walks = []
    num_walks_per_rw = 1 + walk_length + 1 - context_size
    for j in range(num_walks_per_rw):
        walks.append(rw[:, j:j + context_size])
    return torch.cat(walks, dim=0)

def neg_sample(adj, walks_per_node, walk_length, context_size, num_negative_samples=1):
    """Generate negative samples using random node sequences."""
    nodes = torch.arange(adj.shape[0]).repeat(walks_per_node * num_negative_samples)
    rw = torch.randint(adj.shape[0], (nodes.size(0), walk_length * num_negative_samples))
    rw = torch.cat([nodes.view(-1, 1), rw], dim=-1)

    walks = []
    num_walks_per_rw = 1 + walk_length + 1 - context_size
    for j in range(num_walks_per_rw):
        walks.append(rw[:, j:j + context_size])
    return torch.cat(walks, dim=0)

def get_graph_contrastive_loss(pos_rw, neg_rw, embedding, embedding_dim, mapping=None):
    """Compute graph contrastive loss using positive and negative random walks."""
    pos_rw = F.embedding(pos_rw.view(-1), mapping.view(-1, 1)).view(pos_rw.size())
    neg_rw = F.embedding(neg_rw.view(-1), mapping.view(-1, 1)).view(neg_rw.size())
    
    # Positive loss
    start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
    h_start = F.embedding(start, embedding).view(pos_rw.size(0), 1, embedding_dim)
    h_rest = F.embedding(rest.view(-1), embedding).view(pos_rw.size(0), -1, embedding_dim)
    out = (h_start * h_rest).sum(dim=-1)
    pos_loss = torch.logsumexp(out, dim=-1)

    # Negative loss
    start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
    h_start = F.embedding(start, embedding).view(neg_rw.size(0), 1, embedding_dim)
    h_rest = F.embedding(rest.view(-1), embedding).view(neg_rw.size(0), -1, embedding_dim)
    out = (h_start * h_rest).sum(dim=-1)
    neg_loss = torch.logsumexp(out, dim=-1)
    neg_loss = torch.logsumexp(torch.cat((neg_loss.view(-1, 1), pos_loss.view(-1, 1)), dim=-1), dim=-1)

    return -torch.mean(pos_loss - neg_loss)
