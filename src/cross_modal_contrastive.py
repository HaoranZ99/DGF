import torch.nn.functional as F
import torch

class MMS_loss(torch.nn.Module):
    """Multi-Modal Similarity loss with margin."""
    def __init__(self):
        super(MMS_loss, self).__init__()

    def forward(self, S, margin=0.001):
        deltas = margin * torch.eye(S.size(0)).to(S.device)
        S = S - deltas

        target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target)
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        return I2C_loss + C2I_loss

def compute_cross_modal_contrastive_loss(image_embd, text_embd, graph_embd, loss_fn=MMS_loss(), margin=0.001):
    """Compute cross-modal contrastive loss between three modalities."""
    sim_image_text = torch.matmul(image_embd, text_embd.t())
    sim_image_graph = torch.matmul(image_embd, graph_embd.t())
    sim_graph_text = torch.matmul(graph_embd, text_embd.t())
    
    return loss_fn(sim_image_text, margin) + loss_fn(sim_image_graph, margin) + loss_fn(sim_graph_text, margin)
