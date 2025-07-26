import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphClusteringModel(nn.Module):
    def __init__(self, d1, d2, hidden_dim, alpha, beta, num_layers, device):
        super(GraphClusteringModel, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_layers = num_layers
        self.device = device

        # Define separate linear transformations for X1 and X2
        self.linear1 = nn.Linear(d1, hidden_dim)
        self.linear2 = nn.Linear(d2, hidden_dim)

    def symmetric_softmax(self, ZM):
        n, d = ZM.shape
        scale = 1.0 / torch.sqrt(torch.tensor(n, dtype=torch.float32))

        similarity_matrix = torch.mm(ZM.T, ZM) * scale

        # Compute row and column-wise softmax
        exp_similarity = torch.exp(similarity_matrix)
        row_sum = torch.sqrt(torch.sum(exp_similarity, dim=1, keepdim=True))
        col_sum = torch.sqrt(torch.sum(exp_similarity, dim=0, keepdim=True))

        # Normalize by row and column sums
        S = exp_similarity / (row_sum @ col_sum)
        return S

    def forward(self, X1, X2, NAM):

        ZM1 = self.linear1(X1)
        ZM2 = self.linear2(X2)

        ZM1 = F.normalize(ZM1, p=2, dim=1)
        ZM2 = F.normalize(ZM2, p=2, dim=1)
        ZM = (ZM1 + ZM2) / 2

        with torch.no_grad():
            SM1 = self.symmetric_softmax(ZM1)
            SM2 = self.symmetric_softmax(ZM2)
            SM_combined = (SM1 + SM2) / 2

        alpha_term = self.alpha / (self.alpha + 1)
        beta_term = self.beta / (self.beta + 1)  # m = 2

        # Initialize the summations
        NAM = NAM.to(self.device)
        sum_alpha = None
        sum_beta = torch.zeros_like(SM_combined)

        part_alpha = ZM.clone()
        current = ZM.clone()
        for _ in range(self.num_layers):
            current = alpha_term * torch.sparse.mm(NAM, current)
            part_alpha = part_alpha + current

        # Compute the second summation over t for beta
        current_power_beta = torch.eye(ZM.size(1)).to(self.device)
        for _ in range(self.num_layers):
            sum_beta = sum_beta + current_power_beta
            current_power_beta = current_power_beta @ (beta_term * SM_combined)
            # current_power_beta = current_power_beta  # ablation: no latter part
        
        HM = (1 / ((self.alpha + 1) * (self.beta + 1))) * (part_alpha @ sum_beta)
        HM = F.normalize(HM, p=2, dim=1)

        del part_alpha, sum_beta, current, current_power_beta
        # torch.cuda.empty_cache()

        return ZM1, ZM2, HM
