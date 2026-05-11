import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OptimizedConvTranspose3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding, groups, dilation):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        ctx.dilation = dilation
        
        # Use PyTorch's implementation for correctness
        output = F.conv_transpose3d(
            input, weight, bias, stride, padding, output_padding, groups, dilation
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        groups = ctx.groups
        dilation = ctx.dilation
        
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = F.conv3d(
                grad_output, weight, None, stride, padding, dilation, groups
            )
            
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv_transpose3d_weight(
                input, weight.shape, grad_output, stride, padding, 
                output_padding, dilation, groups
            )
            
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3, 4))
            
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class FusedPoolClampSoftmaxMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pool_size, clamp_min, clamp_max):
        # Save for backward
        ctx.pool_size = pool_size
        ctx.clamp_min = clamp_min
        ctx.clamp_max = clamp_max
        
        # Step 1: Average pooling
        pooled = F.avg_pool3d(input, pool_size)
        
        # Step 2: Clamping
        clamped = torch.clamp(pooled, clamp_min, clamp_max)
        
        # Step 3: Softmax
        softmaxed = F.softmax(clamped, dim=1)
        
        # Step 4: Multiplication
        output = softmaxed * 2.0
        
        # Save intermediate results for backward
        ctx.save_for_backward(input, pooled, clamped, softmaxed)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, pooled, clamped, softmaxed = ctx.saved_tensors
        pool_size = ctx.pool_size
        clamp_min = ctx.clamp_min
        clamp_max = ctx.clamp_max
        
        # Backward for multiplication
        grad_softmax = grad_output * 2.0
        
        # Backward for softmax - efficient vectorized implementation
        softmax_sum = (softmaxed * grad_softmax).sum(dim=1, keepdim=True)
        grad_clamped = softmaxed * (grad_softmax - softmax_sum)
        
        # Backward for clamping
        grad_pooled = grad_clamped.clone()
        mask = (pooled < clamp_min) | (pooled > clamp_max)
        grad_pooled[mask] = 0
        
        # Backward for average pooling
        batch_size, channels, pooled_depth, pooled_height, pooled_width = pooled.shape
        depth, height, width = input.shape[2], input.shape[3], input.shape[4]
        
        # Use PyTorch's built-in functionality for gradient calculation
        grad_input = torch.zeros_like(input)
        
        # Distribute gradients evenly across the pooling window
        pool_size_cube = pool_size ** 3
        scale_factor = 1.0 / pool_size_cube
        
        for b in range(batch_size):
            for c in range(channels):
                for pd in range(pooled_depth):
                    for ph in range(pooled_height):
                        for pw in range(pooled_width):
                            d_start = pd * pool_size
                            h_start = ph * pool_size
                            w_start = pw * pool_size
                            
                            grad_val = grad_pooled[b, c, pd, ph, pw] * scale_factor
                            grad_input[b, c, 
                                      d_start:d_start+pool_size, 
                                      h_start:h_start+pool_size, 
                                      w_start:w_start+pool_size] = grad_val
        
        return grad_input, None, None, None

class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, average pooling, clamping, softmax, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        
        # Initialize weight and bias for ConvTranspose3d
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        # Convert to channels_last format for better memory access patterns
        x = x.contiguous(memory_format=torch.channels_last_3d)
        
        # Use mixed precision where available
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Step 1: Perform transposed convolution using our optimized function
            x = OptimizedConvTranspose3d.apply(
                x, self.weight, self.bias, 
                self.stride, self.padding, self.output_padding, 
                1, (1, 1, 1)  # groups=1, dilation=(1,1,1)
            )
            
            # Steps 2-5: Use fused custom function for pooling, clamping, softmax, and multiplication
            x = FusedPoolClampSoftmaxMul.apply(
                x, self.pool_kernel_size, self.clamp_min, self.clamp_max
            )
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]