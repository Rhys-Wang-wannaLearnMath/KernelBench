import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Your optimized implementation here that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features  
        scaling_factor (float): Scaling factor to apply
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # Direct parameter access for maximum control
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = scaling_factor
        
        # Initialize parameters exactly as nn.Linear does
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Cache for transposed weight with version tracking
        self.weight_t = None
        self.weight_version = -1
        
        # We'll initialize the scaling tensor on first forward pass
        self.scaling_tensor = None
        self.initialized = False
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # One-time initialization on first forward pass
        if not self.initialized:
            # Initialize scaling tensor with proper device and dtype
            self.scaling_tensor = torch.tensor(self.scaling_factor, 
                                             device=x.device, 
                                             dtype=x.dtype)
            self.initialized = True
            
        # Lazily transpose weight matrix only when needed with minimal checking
        if self.weight_t is None or self.weight._version != self.weight_version:
            self.weight_t = self.weight.t().contiguous()
            self.weight_version = self.weight._version
        
        # Use addmm for optimized matrix multiplication (bias + x @ weight.T)
        # This fuses the matrix multiplication and bias addition into one operation
        linear_output = torch.addmm(self.bias, x, self.weight_t)
        
        # Apply sigmoid activation
        sigmoid_output = torch.sigmoid(linear_output)
        
        # Use addcmul for fused scaling and residual addition
        # This fuses the multiplication and addition: linear_output + sigmoid_output * scaling_factor
        result = torch.addcmul(linear_output, sigmoid_output, self.scaling_tensor)
        
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_size = 1024
hidden_size = 512
scaling_factor = 2.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [input_size, hidden_size, scaling_factor]