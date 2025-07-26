import numpy as np
from sklearn import metrics
from munkres import Munkres

def calculate_clustering_metrics(true_label, pred_label):
    """
    Calculate clustering performance metrics with optimal label mapping.
    
    Args:
        true_label: Ground truth labels
        pred_label: Predicted cluster labels
        
    Returns:
        tuple: (accuracy, nmi, f1_macro, adjusted_rand_score, completeness_score)
    """
    # Label-independent metrics
    nmi = metrics.normalized_mutual_info_score(true_label, pred_label)
    adjscore = metrics.adjusted_rand_score(true_label, pred_label)
    cs = metrics.completeness_score(true_label, pred_label)

    # Optimal label mapping using Hungarian algorithm
    num_classes = max(true_label) + 1
    cost_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(num_classes):
        true_class_indices = [idx for idx, label in enumerate(true_label) if label == i]
        for j in range(num_classes):
            overlap = len([idx for idx in true_class_indices if pred_label[idx] == j])
            cost_matrix[i, j] = -overlap  # Negative for maximization

    # Find optimal assignment
    munkres_solver = Munkres()
    indexes = munkres_solver.compute(cost_matrix.tolist())
    label_mapping = {original: mapped for mapped, original in indexes}

    # Remap predicted labels
    new_pred_label = np.zeros_like(pred_label)
    for i, pred in enumerate(pred_label):
        if pred in label_mapping:
            new_pred_label[i] = label_mapping[pred]

    # Calculate accuracy and F1 with remapped labels
    acc = metrics.accuracy_score(true_label, new_pred_label)
    f1_macro = metrics.f1_score(true_label, new_pred_label, average='macro')

    return acc, nmi, f1_macro, adjscore, cs