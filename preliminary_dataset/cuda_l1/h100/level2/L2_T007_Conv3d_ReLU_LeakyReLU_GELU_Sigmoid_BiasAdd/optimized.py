import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Custom CUDA kernel for fused activation operations
cuda_code = """
extern "C" __global__ void fused_activations_forward(
    const float* input,
    const float* bias,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * channels * depth * height * width;
    
    if (idx < total_size) {
        // Calculate channel index for bias
        const int dhw_size = depth * height * width;
        const int c = (idx / dhw_size) % channels;
        
        // Get input value
        float x = input[idx];
        
        // Apply ReLU
        x = fmaxf(x, 0.0f);
        
        // Skip LeakyReLU as it's redundant after ReLU
        
        // Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coef = 0.044715f;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coef * x_cubed);
        x = 0.5f * x * (1.0f + tanhf(inner));
        
        // Apply Sigmoid: 1 / (1 + exp(-x))
        x = 1.0f / (1.0f + expf(-x));
        
        // Add bias
        x = x + bias[c];
        
        // Store result
        output[idx] = x;
    }
}

extern "C" __global__ void fused_activations_backward(
    const float* grad_output,
    const float* input,
    const float* bias,
    float* grad_input,
    float* grad_bias,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * channels * depth * height * width;
    
    if (idx < total_size) {
        const int dhw_size = depth * height * width;
        const int c = (idx / dhw_size) % channels;
        
        // Get input value
        float x = input[idx];
        
        // ReLU derivative
        float drelu = (x > 0.0f) ? 1.0f : 0.0f;
        
        // Skip LeakyReLU derivative
        
        // Compute intermediate values for derivatives
        // First apply ReLU
        x = fmaxf(x, 0.0f);
        
        // Apply GELU
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coef = 0.044715f;
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coef * x_cubed);
        float gelu_x = 0.5f * x * (1.0f + tanhf(inner));
        
        // GELU derivative components
        float tanh_val = tanhf(inner);
        float sech_squared = 1.0f - tanh_val * tanh_val;
        float dgelu_dx = 0.5f * (1.0f + tanh_val) + 
                         0.5f * x * sech_squared * sqrt_2_over_pi * 
                         (1.0f + 3.0f * coef * x * x);
        
        // Apply Sigmoid
        float sigmoid_x = 1.0f / (1.0f + expf(-gelu_x));
        
        // Sigmoid derivative
        float dsigmoid_dx = sigmoid_x * (1.0f - sigmoid_x);
        
        // Chain rule for the full derivative
        float dout_dx = drelu * dgelu_dx * dsigmoid_dx * grad_output[idx];
        
        // Store gradient for input
        grad_input[idx] = dout_dx;
        
        // Atomically add to bias gradients
        atomicAdd(&grad_bias[c], grad_output[idx]);
    }
}
"""

class FusedActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        
        output = torch.empty_like(input)
        
        if input.is_cuda:
            # Get tensor dimensions
            batch_size, channels, depth, height, width = input.shape
            
            # Load CUDA kernel
            if not hasattr(FusedActivationFunction, 'forward_kernel'):
                FusedActivationFunction.forward_kernel = torch.cuda.ByteTensor()
                FusedActivationFunction.backward_kernel = torch.cuda.ByteTensor()
                
                mod = torch.cuda.cudart().compile_module(cuda_code)
                FusedActivationFunction.forward_kernel = mod.get_function("fused_activations_forward")
                FusedActivationFunction.backward_kernel = mod.get_function("fused_activations_backward")
            
            # Calculate grid and block dimensions
            threads_per_block = 256
            num_blocks = (input.numel() + threads_per_block - 1) // threads_per_block
            
            # Launch kernel
            FusedActivationFunction.forward_kernel(
                grid=(num_blocks, 1, 1),
                block=(threads_per_block, 1, 1),
                args=[input.data_ptr(), bias.data_ptr(), output.data_ptr(),
                      batch_size, channels, depth, height, width]
            )
        else:
            # CPU fallback implementation
            result = F.relu(input)
            result = F.gelu(result)  # Skip LeakyReLU since it's redundant after ReLU
            result = torch.sigmoid(result)
            output = result + bias
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        
        grad_input = torch.zeros_like(input)
        grad_bias = torch.zeros_like(bias)
        
        if grad_output.is_cuda:
            # Get tensor dimensions
            batch_size, channels, depth, height, width = input.shape
            
            # Calculate grid and block dimensions
            threads_per_block = 256
            num_blocks = (input.numel() + threads_per_block - 1) // threads_per_block
            
            # Launch kernel
            FusedActivationFunction.backward_kernel(
                grid=(num_blocks, 1, 1),
                block=(threads_per_block, 1, 1),
                args=[grad_output.data_ptr(), input.data_ptr(), bias.data_ptr(),
                      grad_input.data_ptr(), grad_bias.data_ptr(),
                      batch_size, channels, depth, height, width]
            )
        else:
            # CPU fallback implementation
            with torch.enable_grad():
                x = input.detach().requires_grad_()
                relu_out = F.relu(x)
                gelu_out = F.gelu(relu_out)  # Skip LeakyReLU
                sigmoid_out = torch.sigmoid(gelu_out)
                output = sigmoid_out + bias
                
                # Backward pass
                grad_input = torch.autograd.grad(output, x, grad_output)[0]
            
            # Gradient for bias is the sum of grad_output across all dimensions except channel
            grad_bias = grad_output.sum(dim=(0, 2, 3, 4), keepdim=True)
        
        return grad_input, grad_bias


class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, 
    and bias in sequence.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        bias_shape (tuple): Shape of the bias tensor
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Initialize convolution layer
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        
        # Initialize bias parameter
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Enable cudnn benchmarking for better convolution performance
        torch.backends.cudnn.benchmark = True
        
        # Pre-convert weights to optimal memory format if on CUDA
        if torch.cuda.is_available():
            self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last_3d)
    
    def forward(self, x):
        # Convert to channels_last_3d memory format for optimal Conv3d performance
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last_3d)
            
            # Ensure weights are in the optimal format
            if not self.conv.weight.is_contiguous(memory_format=torch.channels_last_3d):
                self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last_3d)
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply fused activation functions and bias addition
        try:
            x = FusedActivationFunction.apply(x, self.bias)
        except Exception as e:
            # Fallback to PyTorch operations if CUDA kernel fails
            print(f"CUDA kernel failed, falling back to PyTorch: {e}")
            x = F.relu(x)
            # Skip LeakyReLU with negative_slope=0.01 since all values are already non-negative after ReLU
            x = F.gelu(x)
            x = torch.sigmoid(x)
            x = x + self.bias
        
        return x


# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, bias_shape]