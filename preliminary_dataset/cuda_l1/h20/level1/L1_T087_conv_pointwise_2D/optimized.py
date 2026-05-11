import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation with highly optimized implementation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize weight parameter with the same shape as nn.Conv2d for compatibility
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using a highly optimized implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        batch_size, _, height, width = x.shape
        
        # Directly reshaping and transposing weight (REMOVE caching)
        weight_2d = self.weight.view(self.out_channels, self.in_channels)
        weight_t = weight_2d.t().contiguous()
        
        # Reshape input: [B, C_in, H, W] -> [B*H*W, C_in]
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, self.in_channels)
        
        # Ensure x_flat is contiguous for optimal matrix multiplication
        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()
        
        # Optimized matrix multiplication: [B*H*W, C_in] @ [C_in, C_out] -> [B*H*W, C_out]
        output = torch.mm(x_flat, weight_t)
        
        # Add bias if needed (in-place operation for efficiency)
        if self.bias is not None:
            output.add_(self.bias)
        
        # Reshape back: [B*H*W, C_out] -> [B, H, W, C_out] -> [B, C_out, H, W]
        output = output.view(batch_size, height, width, self.out_channels).permute(0, 3, 1, 2)
        
        return output

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels]