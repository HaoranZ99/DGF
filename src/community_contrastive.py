import torch
import torch.nn.functional as F
import math

def cluster_loss(H, centroids, labels, positive_threshold_percent=0.2):
    """
    Calculate cluster loss based on similarity to centroids.
    
    Args:
        H: Normalized features (n, d)
        centroids: K-Means centroids (K, d)  
        labels: Cluster assignments (n,)
        positive_threshold_percent: Percentage of top samples per cluster for numerator
    
    Returns:
        torch.Tensor: Calculated loss value
    """
    n_samples_total = H.shape[0]
    K = centroids.shape[0]

    if n_samples_total == 0 or K == 0:
        return torch.tensor(0.0, device=H.device, dtype=H.dtype)

    total_loss_sum = 0.0

    for k_cluster_idx in range(K):
        current_centroid_k_vec = centroids[k_cluster_idx]
        indices_of_samples_in_k_all = (labels == k_cluster_idx).nonzero(as_tuple=True)[0]
        num_samples_in_cluster_k_all = len(indices_of_samples_in_k_all)

        if num_samples_in_cluster_k_all == 0:
            continue

        H_samples_in_k_all = H[indices_of_samples_in_k_all]

        # Calculate numerator: top-k positive samples
        sum_exp_pos_scores_k = torch.tensor(0.0, device=H.device, dtype=H.dtype)
        similarities_to_own_centroid = torch.matmul(H_samples_in_k_all, current_centroid_k_vec)

        k_for_topk = math.ceil(num_samples_in_cluster_k_all * positive_threshold_percent)
        k_for_topk = min(max(k_for_topk, 0), num_samples_in_cluster_k_all)
        
        if k_for_topk > 0:
            _, top_indices_within_cluster = torch.topk(similarities_to_own_centroid, k=int(k_for_topk))
            scores_pos_k_individual = similarities_to_own_centroid[top_indices_within_cluster]
            sum_exp_pos_scores_k = torch.sum(torch.exp(scores_pos_k_individual))
        
        if sum_exp_pos_scores_k <= 0:
            continue

        # Calculate denominator: similarities to all centroids
        sim_all_samples_k_with_all_centroids = torch.matmul(H_samples_in_k_all, centroids.T)
        sum_exp_sim_hm_with_all_centroids_per_sample = torch.sum(torch.exp(sim_all_samples_k_with_all_centroids), dim=1)
        simplified_denominator_k = torch.sum(sum_exp_sim_hm_with_all_centroids_per_sample)

        if simplified_denominator_k <= 0:
            continue
            
        # Calculate loss for cluster k
        ratio = sum_exp_pos_scores_k / simplified_denominator_k
        cluster_k_loss = -torch.log(ratio)
        total_loss_sum += cluster_k_loss

    return total_loss_sum / n_samples_total if n_samples_total > 0 else torch.tensor(0.0, device=H.device, dtype=H.dtype)
