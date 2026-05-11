import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool): Whether to use bias
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and bias similar to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters using same method as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-compute constants as simple tensors (following No3's successful approach)
        self._half = torch.tensor(0.5)
        self._neg_one = torch.tensor(-1.0)
        self._pos_one = torch.tensor(1.0)
        self._constants_device = None
    
    def forward(self, x):
        """
        Optimized forward pass with maximum memory efficiency
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Cache device for efficiency
        device = x.device
        
        # Move constants to correct device if needed (following No3's successful pattern)
        if self._constants_device != device:
            self._half = self._half.to(device, non_blocking=True)
            self._neg_one = self._neg_one.to(device, non_blocking=True) 
            self._pos_one = self._pos_one.to(device, non_blocking=True)
            self._constants_device = device
        
        # Retrieve and transpose weight every time (removed caching mechanism)
        weight = self.weight
        weight_t = weight.t().contiguous()
        
        bias = self.bias
        half = self._half
        neg_one = self._neg_one
        pos_one = self._pos_one
        
        # Linear transformation using optimal GEMM operation
        if bias is not None:
            output = torch.addmm(bias, x, weight_t)
        else:
            output = torch.mm(x, weight_t)
        
        # Swish activation using optimized SiLU function
        output = torch.nn.functional.silu(output)
        
        # Divide by 2.0 using multiplication (faster than division)
        output.mul_(half)
        
        # First clamp operation (in-place)
        output.clamp_(neg_one, pos_one)
        
        # Tanh activation (in-place)
        output.tanh_()
        
        # Final clamp operation (in-place, kept for functional equivalence)
        output.clamp_(neg_one, pos_one)
        
        return output

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features]