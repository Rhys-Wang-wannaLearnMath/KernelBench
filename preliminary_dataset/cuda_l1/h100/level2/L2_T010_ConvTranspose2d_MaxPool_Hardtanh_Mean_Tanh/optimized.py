import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, followed by max pooling, hardtanh activation, mean operation, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Initialize weight and bias parameters directly for optimal control
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # Store parameters for operations
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        
        # Initialize weights and biases using the same approach as nn.ConvTranspose2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Pre-compute all output dimensions for maximum efficiency
        self.input_h, self.input_w = height, width  # From reference implementation
        self.conv_out_h = (self.input_h - 1) * stride - 2 * padding + kernel_size
        self.conv_out_w = (self.input_w - 1) * stride - 2 * padding + kernel_size
        self.maxpool_out_h = (self.conv_out_h - maxpool_kernel_size) // maxpool_stride + 1
        self.maxpool_out_w = (self.conv_out_w - maxpool_kernel_size) // maxpool_stride + 1
        self.spatial_size = self.maxpool_out_h * self.maxpool_out_w
        
        # Convert weights to channels_last format for better memory access
        self.weight.data = self.weight.data.contiguous(memory_format=torch.channels_last)
        self.bias.data = self.bias.data.contiguous()

    def forward(self, x):
        batch_size = x.size(0)
        
        # Ensure input tensor is contiguous and optimally laid out
        x = x.to(memory_format=torch.channels_last)
        
        # Step 1: ConvTranspose2d with direct functional call for optimal performance
        x = F.conv_transpose2d(
            x, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding
        )
        
        # Step 2: MaxPool2d with optimized parameters
        x = F.max_pool2d(
            x,
            kernel_size=self.maxpool_kernel_size,
            stride=self.maxpool_stride,
            ceil_mode=False
        )
        
        # Step 3: In-place Hardtanh to minimize memory allocation
        x.clamp_(min=self.hardtanh_min, max=self.hardtanh_max)
        
        # Step 4: Optimized mean operation using pre-computed dimensions
        # Use view instead of reshape for better performance when possible
        x = x.view(batch_size, out_channels, self.spatial_size)
        x = x.mean(dim=2).view(batch_size, out_channels, 1, 1)
        
        # Step 5: Tanh activation
        x = torch.tanh(x)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]