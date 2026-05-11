import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        scale_shape (tuple): Shape of the scaling factor
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Keep the same modules as the reference implementation
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        
        # Cache tensor properties and references for faster access
        self.has_bias = self.gemm.bias is not None

    def forward(self, x):
        """
        Ultra-optimized forward pass with maximum CUDA efficiency
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Fused GEMM + scaling operation without cached matrices
        weight_t = self.gemm.weight.t().contiguous()
        scale_expanded = self.scale.view(1, -1)
        scaled_weight_t = (weight_t * scale_expanded).contiguous()
        
        if self.has_bias:
            scaled_bias = (self.gemm.bias * self.scale).contiguous()
            out = torch.addmm(scaled_bias, x, scaled_weight_t)
        else:
            out = torch.mm(x, scaled_weight_t)
        
        # Apply batch normalization using PyTorch's optimized implementation
        out = self.bn(out)
        
        return out

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
scale_shape = (out_features,)

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features, scale_shape]