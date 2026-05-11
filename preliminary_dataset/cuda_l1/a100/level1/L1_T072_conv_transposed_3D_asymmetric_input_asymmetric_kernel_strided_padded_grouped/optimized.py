import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with asymmetric input and kernel, and optional stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple of ints): Size of the convolution kernel in the form (kernel_size_depth, kernel_size_height, kernel_size_width).
        stride (tuple of ints, optional): Stride of the convolution in the form (stride_depth, stride_height, stride_width). Defaults to (1, 1, 1).
        padding (tuple of ints, optional): Padding applied to the input in the form (padding_depth, padding_height, padding_width). Defaults to (0, 0, 0).
        output_padding (tuple of ints, optional): Additional size added to one side of the output shape. Defaults to (0, 0, 0).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), 
                 padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.groups = groups
        
        # Create the standard PyTorch implementation for correctness and fallback
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding, 
            groups=groups, bias=bias
        )
        
        # Precompute values for optimization
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups
        
        # Precompute channel indices for each group to avoid redundant calculations
        self.in_channel_indices = [(g * self.in_channels_per_group, (g + 1) * self.in_channels_per_group) 
                                   for g in range(self.groups)]
        self.out_channel_indices = [(g * self.out_channels_per_group, (g + 1) * self.out_channels_per_group) 
                                    for g in range(self.groups)]
        
        # Cache for optimized weight tensors
        self.optimized_weight = None
        self.weight_version = None
        self.weight_slices = None
        
        # Flag to enable/disable optimization
        self.use_optimized = True
        
    def _optimize_memory_layout(self, x):
        """Optimize tensor memory layout for GPU operations"""
        # Ensure tensor is contiguous
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Try to use channels_last_3d memory format if available
        if x.is_cuda and hasattr(torch, 'channels_last_3d'):
            try:
                x = x.to(memory_format=torch.channels_last_3d)
            except:
                pass
                
        return x
    
    def _prepare_optimized_weight(self):
        """Prepare and cache optimized weight tensor"""
        weight = self.conv_transpose3d.weight
        current_version = weight._version
        
        # Check if we need to update the cached weight
        if (self.optimized_weight is None or 
            self.weight_version != current_version or 
            self.optimized_weight.shape != weight.shape):
            
            # Optimize memory layout for weight tensor
            self.optimized_weight = self._optimize_memory_layout(weight)
            
            # Pre-slice weights for each group
            self.weight_slices = []
            for g in range(self.groups):
                start_idx = g * self.in_channels_per_group
                end_idx = (g + 1) * self.in_channels_per_group
                self.weight_slices.append(self.optimized_weight[start_idx:end_idx])
                
            self.weight_version = current_version
        
        return self.weight_slices
        
    def _optimized_grouped_conv(self, x):
        """Optimized implementation for grouped convolution"""
        # Get optimized weight slices
        weight_slices = self._prepare_optimized_weight()
        
        # Get bias tensor
        bias = self.conv_transpose3d.bias
        
        # Create streams for parallel execution if available
        streams = []
        if torch.cuda.is_available():
            streams = [torch.cuda.Stream() for _ in range(self.groups)]
        
        # Pre-allocate list for group outputs
        group_outputs = [None] * self.groups
        
        # Process each group
        for g in range(self.groups):
            # Use CUDA stream if available
            if streams:
                torch.cuda.set_stream(streams[g])
            
            # Extract input slice for this group
            start_idx = g * self.in_channels_per_group
            end_idx = (g + 1) * self.in_channels_per_group
            x_g = x[:, start_idx:end_idx].contiguous()
            
            # Get pre-sliced weight for this group
            weight_g = weight_slices[g]
            
            # Extract bias slice for this group if bias is used
            bias_g = None
            if bias is not None:
                bias_start = g * self.out_channels_per_group
                bias_end = (g + 1) * self.out_channels_per_group
                bias_g = bias[bias_start:bias_end]
            
            # Apply transposed convolution for this group
            out_g = F.conv_transpose3d(
                x_g, weight_g, bias_g,
                stride=self.stride, 
                padding=self.padding,
                output_padding=self.output_padding, 
                groups=1  # groups=1 since we're already handling groups manually
            )
            
            group_outputs[g] = out_g
        
        # Synchronize streams before concatenation
        if streams:
            torch.cuda.synchronize()
        
        # Concatenate group outputs along channel dimension
        return torch.cat(group_outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Early exit for non-CUDA tensors or non-grouped convolutions
        if not x.is_cuda or self.groups <= 1:
            return self.conv_transpose3d(x)
        
        # Try optimized implementation for grouped convolution
        if self.use_optimized:
            try:
                # Optimize memory layout for input tensor
                x_optimized = self._optimize_memory_layout(x)
                
                # Apply optimized grouped convolution
                return self._optimized_grouped_conv(x_optimized)
            except Exception:
                # If optimization fails, disable it for future calls and fall back to standard implementation
                self.use_optimized = False
        
        # Fallback to standard PyTorch implementation
        # Try to optimize weight tensor memory layout for standard implementation
        if hasattr(torch, 'channels_last_3d'):
            try:
                if not self.conv_transpose3d.weight.is_contiguous(memory_format=torch.channels_last_3d):
                    self.conv_transpose3d.weight.data = self.conv_transpose3d.weight.data.to(memory_format=torch.channels_last_3d)
            except:
                pass
        
        return self.conv_transpose3d(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5, 7)
depth = 16
height = 32
width = 64
stride = (2, 2, 2)
padding = (1, 2, 3)
output_padding = (1, 1, 1)
groups = 4

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, groups]