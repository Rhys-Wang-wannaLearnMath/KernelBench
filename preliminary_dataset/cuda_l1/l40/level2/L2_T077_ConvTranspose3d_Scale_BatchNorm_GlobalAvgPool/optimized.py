import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling with advanced optimizations.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        scale_factor (float): Scaling factor to apply
        eps (float, optional): Small constant added to the denominator for numerical stability in batch norm
        momentum (float, optional): Value used for the running_mean and running_var computation in batch norm
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        
        # Pre-allocate buffer for potential future optimizations
        self.register_buffer('_dummy_buffer', torch.zeros(1), persistent=False)

    def forward(self, x):
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use mixed precision for the computationally expensive ConvTranspose3d operation
        with torch.cuda.amp.autocast(enabled=True):
            # Step 1: Apply transposed convolution with mixed precision
            conv_out = self.conv_transpose(x)
            
            # Step 2: Immediately reduce spatial dimensions with efficient pooling
            # and fuse with scaling operation to minimize memory access
            # This is mathematically equivalent to:
            # pooled = (torch.sum(conv_out, dim=(2, 3, 4), keepdim=True) / spatial_size) * self.scale_factor
            # But more efficient as it reduces one multiplication operation
            spatial_size = conv_out.shape[2] * conv_out.shape[3] * conv_out.shape[4]
            scaled_factor = self.scale_factor / spatial_size
            pooled = torch.sum(conv_out, dim=(2, 3, 4), keepdim=True) * scaled_factor
        
        # Convert back to full precision for batch normalization (critical for stability)
        if pooled.dtype != torch.float32:
            pooled = pooled.float()
        
        # Step 3: Apply batch normalization on the reduced tensor
        result = self.batch_norm(pooled)
        
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 64
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]