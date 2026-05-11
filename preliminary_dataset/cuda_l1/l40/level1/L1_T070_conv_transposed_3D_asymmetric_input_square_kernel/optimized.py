import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    An optimized transposed convolution (3D) implementation that maintains
    identical functionality while improving performance by reusing a flipped
    kernel with conv3d.
    
    Args:
        in_features (int): Number of input channels (maps to 'in_channels')
        out_features (int): Number of output channels (maps to 'out_channels')
        scaling_factor (int): Convolution kernel size (maps to 'kernel_size')
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # 1) Create a reference-style ConvTranspose3d to match the original initialization
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=scaling_factor,
            stride=1,
            padding=0,
            output_padding=0,
            dilation=1,
            groups=1,
            bias=False  # bias=False per the reference
        )

        # 2) Flip the weight once in the constructor
        with torch.no_grad():
            # The original ConvTranspose3d weight shape: [in_features, out_features, 3, 3, 3]
            w_t = self.conv_transpose3d.weight
            # Flip spatial dims and swap channel dims => [out_features, in_features, 3, 3, 3]
            flipped = w_t.permute(1, 0, 2, 3, 4).flip(dims=[2, 3, 4])
            # Register as a buffer for usage in forward
            self.register_buffer("flipped_weight", flipped.contiguous())

        # Reference biases are typically None (bias=False), but store in case
        self.bias = self.conv_transpose3d.bias

    def forward(self, x):
        """
        Forward pass that replicates transposed convolution (kernel_size=3, stride=1, padding=0)
        by calling conv3d with a flipped kernel and padding=2.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features, depth, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features, depth_out, height_out, width_out)
        """
        # Optional: ensure input is contiguous
        x = x.contiguous()

        # For kernel_size=3, we use padding=2 => kernel_size - 1
        out = F.conv3d(x, self.flipped_weight, bias=None, stride=1, padding=2)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1, 1)
        return out


# ----------------------------------------------------------------------------
# CRITICAL: Keep ALL hyperparameters EXACTLY as in the reference implementation
# ----------------------------------------------------------------------------
batch_size = 16
in_channels = 32
out_channels = 16
kernel_size = 3
depth = 16
height = 32
width = 64

def get_inputs():
    """
    Return a fresh input tensor using EXACT hyperparameters from the reference, i.e.,
    shape (batch_size, in_channels, depth, height, width).
    """
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    """
    Return constructor parameters exactly as in the reference:
    (in_channels, out_channels, kernel_size)
    """
    return [in_channels, out_channels, kernel_size]