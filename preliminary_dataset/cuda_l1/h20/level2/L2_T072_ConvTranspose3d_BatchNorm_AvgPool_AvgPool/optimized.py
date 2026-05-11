import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvTranspose3dCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding, groups, dilation):
        # This is a placeholder for the custom CUDA kernel
        # For now, we'll use the PyTorch implementation
        output = F.conv_transpose3d(
            input, weight, bias, 
            stride=stride, 
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Not needed for inference
        return None, None, None, None, None, None, None, None

class ModelNew(nn.Module):
    """
    An optimized model that performs a 3D transposed convolution, followed by batch normalization,
    two average pooling layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        
        # Create reference modules to ensure identical initialization
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)
        
        # Store configuration for functional API calls
        self.stride = stride
        self.padding = padding
        self.output_padding = 0
        self.groups = 1
        self.dilation = 1
        self.eps = self.batch_norm.eps
        self.momentum = 0.1  # Default PyTorch momentum
        
        # Enable cudnn benchmark for kernel autotuning
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Try to use torch.compile if available (PyTorch 2.0+)
        self.use_compile = hasattr(torch, 'compile')
        if self.use_compile:
            try:
                self.optimized_forward = torch.compile(self._optimized_forward)
            except:
                self.use_compile = False
    
    def _optimized_forward(self, x, weight, bias, bn_weight, bn_bias, running_mean, running_var):
        # Step 1: ConvTranspose3d using functional API
        x = F.conv_transpose3d(
            x, weight, bias, 
            stride=self.stride, 
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation
        )
        
        # Step 2: BatchNorm3d using functional API
        x = F.batch_norm(
            x,
            running_mean,
            running_var,
            bn_weight,
            bn_bias,
            False,  # Not training
            self.momentum,
            self.eps
        )
        
        # Step 3: Fused pooling - replace two consecutive AvgPool3d(kernel_size=2) 
        # with a single AvgPool3d(kernel_size=4, stride=4)
        x = F.avg_pool3d(x, kernel_size=4, stride=4)
        
        return x
        
    def forward(self, x):
        # Use no_grad for inference optimization
        with torch.no_grad():
            device = x.device
            
            # Try using channels_last_3d memory format if available
            try:
                if hasattr(torch.memory_format, 'channels_last_3d'):
                    x = x.to(memory_format=torch.memory_format.channels_last_3d)
                    weight = self.conv_transpose.weight.to(memory_format=torch.memory_format.channels_last_3d)
                else:
                    weight = self.conv_transpose.weight
            except Exception:
                weight = self.conv_transpose.weight
            
            # Extract parameters once to avoid repeated attribute access
            bias = self.conv_transpose.bias
            bn_weight = self.batch_norm.weight
            bn_bias = self.batch_norm.bias
            running_mean = self.batch_norm.running_mean
            running_var = self.batch_norm.running_var
            
            # Make sure all tensors are on the same device
            weight = weight.to(device)
            bias = bias.to(device)
            bn_weight = bn_weight.to(device)
            bn_bias = bn_bias.to(device)
            running_mean = running_mean.to(device)
            running_var = running_var.to(device)
            
            # Use compiled version if available, otherwise use direct implementation
            if self.use_compile:
                try:
                    return self.optimized_forward(x, weight, bias, bn_weight, bn_bias, running_mean, running_var)
                except:
                    pass
            
            # Calculate output shape for pre-allocation
            batch_size, in_channels, in_depth, in_height, in_width = x.shape
            out_depth = (in_depth - 1) * self.stride + self.dilation * (self.conv_transpose.kernel_size[0] - 1) + 1 - 2 * self.padding
            out_height = (in_height - 1) * self.stride + self.dilation * (self.conv_transpose.kernel_size[1] - 1) + 1 - 2 * self.padding
            out_width = (in_width - 1) * self.stride + self.dilation * (self.conv_transpose.kernel_size[2] - 1) + 1 - 2 * self.padding
            
            # Step 1: ConvTranspose3d using functional API
            x = F.conv_transpose3d(
                x, weight, bias, 
                stride=self.stride, 
                padding=self.padding,
                output_padding=self.output_padding,
                groups=self.groups,
                dilation=self.dilation
            )
            
            # Step 2: BatchNorm3d using functional API
            x = F.batch_norm(
                x,
                running_mean,
                running_var,
                bn_weight,
                bn_bias,
                False,  # Not training
                self.momentum,
                self.eps
            )
            
            # Step 3: Fused pooling - replace two consecutive AvgPool3d(kernel_size=2) 
            # with a single AvgPool3d(kernel_size=4, stride=4)
            x = F.avg_pool3d(x, kernel_size=4, stride=4)
            
            return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]