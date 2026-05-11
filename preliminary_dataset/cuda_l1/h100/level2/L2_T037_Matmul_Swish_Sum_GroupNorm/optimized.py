import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearSwishBiasFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        batch_size, in_features = input.shape
        out_features = weight.shape[0]
        
        # Allocate output tensor
        output = torch.empty((batch_size, out_features), device=input.device, dtype=input.dtype)
        
        if input.is_cuda:
            # Define CUDA kernel
            cuda_kernel = '''
            extern "C" __global__ void linear_swish_bias_fused(
                const float* input, const float* weight, const float* bias,
                float* output, int batch_size, int in_features, int out_features) {
                
                // Block and thread indices
                const int tid = threadIdx.x;
                const int bid = blockIdx.x;
                const int bdim = blockDim.x;
                
                // Calculate row index (batch dimension)
                const int row = bid;
                
                // Check if row is valid
                if (row < batch_size) {
                    // Each thread processes multiple output elements
                    for (int col = tid; col < out_features; col += bdim) {
                        float sum = 0.0f;
                        
                        // Compute dot product
                        for (int i = 0; i < in_features; ++i) {
                            sum += input[row * in_features + i] * weight[col * in_features + i];
                        }
                        
                        // Add bias
                        sum += bias[col];
                        
                        // Apply Swish: x * sigmoid(x)
                        const float sigmoid_val = 1.0f / (1.0f + expf(-sum));
                        output[row * out_features + col] = sum * sigmoid_val;
                    }
                }
            }
            '''
            
            # Load and compile the CUDA kernel if not already loaded
            if not hasattr(LinearSwishBiasFused, '_kernel_loaded'):
                import cupy as cp
                LinearSwishBiasFused._kernel = cp.RawKernel(cuda_kernel, 'linear_swish_bias_fused')
                LinearSwishBiasFused._kernel_loaded = True
            
            # Launch the kernel
            threads_per_block = 256
            blocks_per_grid = batch_size
            
            # Use CuPy to launch the kernel
            import cupy as cp
            LinearSwishBiasFused._kernel(
                (blocks_per_grid,), (threads_per_block,),
                (cp.asarray(input), cp.asarray(weight), cp.asarray(bias),
                 cp.asarray(output), batch_size, in_features, out_features)
            )
        else:
            # Fallback to PyTorch implementation
            output = F.linear(input, weight, bias)
            output = torch.sigmoid(output) * output
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        
        # Forward pass to recompute intermediate values
        linear_output = F.linear(input, weight, bias)
        sigmoid_output = torch.sigmoid(linear_output)
        
        # Gradient for Swish: grad_output * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
        swish_grad = sigmoid_output * (1 + linear_output * (1 - sigmoid_output))
        grad_output_times_swish_grad = grad_output * swish_grad
        
        # Gradient for input
        grad_input = grad_output_times_swish_grad @ weight
        
        # Gradient for weight
        grad_weight = grad_output_times_swish_grad.t() @ input
        
        # Gradient for bias
        grad_bias = grad_output_times_swish_grad.sum(0)
        
        return grad_input, grad_weight, grad_bias

class ModelNew(nn.Module):
    """
    An optimized model that performs a matrix multiplication, applies Swish activation,
    sums with a bias term, and normalizes with GroupNorm.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        num_groups (int): Number of groups for GroupNorm
        bias_shape (tuple): Shape of the bias tensor
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        # Use PyTorch's optimized Linear layer
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_linear = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_linear, -bound, bound)
        
        # Bias parameter exactly as in reference
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Use PyTorch's optimized GroupNorm
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        
        # Flag to determine if we can use custom CUDA kernel
        self.use_cuda_kernel = False
        try:
            import cupy
            self.use_cuda_kernel = True
        except ImportError:
            pass
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Try to use our custom fused kernel if CUDA is available
        if x.is_cuda and self.use_cuda_kernel:
            try:
                x = LinearSwishBiasFused.apply(x, self.weight, self.bias_linear)
            except Exception:
                # Fallback to PyTorch implementation
                x = F.linear(x, self.weight, self.bias_linear)
                x = F.silu(x, inplace=True)
        else:
            # Use PyTorch's optimized implementation
            x = F.linear(x, self.weight, self.bias_linear)
            x = F.silu(x, inplace=True)
        
        # Add bias in-place to reduce memory allocation
        x.add_(self.bias)
        
        # Apply group normalization
        x = self.group_norm(x)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 512
out_features = 1024
num_groups = 32
bias_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]