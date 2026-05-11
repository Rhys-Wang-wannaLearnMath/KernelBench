import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    An optimized model that computes Kullback-Leibler Divergence for comparing two distributions.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        """
        Optimized KL divergence computation using direct mathematical formulation
        
        Args:
            predictions (torch.Tensor): Predicted probability distribution
            targets (torch.Tensor): Target probability distribution
            
        Returns:
            torch.Tensor: KL divergence loss (scalar)
        """
        # Ensure contiguous memory layout only if needed
        if not predictions.is_contiguous():
            predictions = predictions.contiguous()
        if not targets.is_contiguous():
            targets = targets.contiguous()
        
        # Direct KL computation: KL(P||Q) = sum(P * log(P/Q))
        # Computing P/Q directly and using torch.xlogy for stability and efficiency
        ratio = targets / predictions
        
        # torch.xlogy handles the case where targets=0 (returns 0)
        kl_terms = torch.xlogy(targets, ratio)
        
        # Efficient reduction: sum over features, then mean over batch
        # Using -1 to specify the last dimension explicitly
        return kl_terms.sum(dim=-1).mean()

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).softmax(dim=-1), torch.randn(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []