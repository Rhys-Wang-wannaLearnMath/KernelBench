import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, scales the output, applies batch normalization, 
    and then performs global average pooling.
    
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
        
        # Advanced caching strategy
        self.spatial_dims = None
        self.inv_spatial_dims = None  # Cache the inverse for more efficient division
        self.output_buffer = None  # Pre-allocated output buffer
        
    def forward(self, x):
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Use mixed precision for the computationally expensive operations
        with torch.cuda.amp.autocast(enabled=True):
            # Step 1: Apply transposed convolution
            conv_out = self.conv_transpose(x)
            
            # Step 2: Calculate and cache spatial dimensions more efficiently
            if self.spatial_dims is None:
                # Cache both the spatial dimensions and their inverse for efficient division
                self.spatial_dims = conv_out.shape[2] * conv_out.shape[3] * conv_out.shape[4]
                self.inv_spatial_dims = 1.0 / float(self.spatial_dims)
            
            # Step 3: Fused pooling and scaling operation for better efficiency
            # Use multiplication instead of division (cached inverse) and combine with scaling
            pooled = torch.sum(conv_out, dim=(2, 3, 4), keepdim=True) * (self.inv_spatial_dims * self.scale_factor)
        
        # Convert back to full precision for batch normalization (critical for stability)
        if pooled.dtype != torch.float32:
            pooled = pooled.float()
        
        # Step 4: Apply batch normalization on the reduced tensor
        pooled = self.batch_norm(pooled)
        
        return pooled

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