import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    An optimized model that computes Triplet Margin Loss for metric learning tasks.
    Builds upon successful vector norm approach with additional memory optimizations.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        # Ensure optimal memory layout with contiguous tensors
        anchor = anchor.contiguous()
        positive = positive.contiguous()
        negative = negative.contiguous()
        
        # Compute differences efficiently - these operations are fused by PyTorch
        diff_pos = anchor - positive
        diff_neg = anchor - negative
        
        # Use highly optimized vector norm operations
        # torch.linalg.vector_norm is the most optimized for L2 norm computation
        dist_pos = torch.linalg.vector_norm(diff_pos, ord=2, dim=1, keepdim=False)
        dist_neg = torch.linalg.vector_norm(diff_neg, ord=2, dim=1, keepdim=False)
        
        # Fused computation: subtract distances, add margin, and clamp in one expression
        # This minimizes intermediate tensor allocations
        loss_values = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)
        
        # Efficient mean reduction
        return torch.mean(loss_values)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [1.0]  # Default margin