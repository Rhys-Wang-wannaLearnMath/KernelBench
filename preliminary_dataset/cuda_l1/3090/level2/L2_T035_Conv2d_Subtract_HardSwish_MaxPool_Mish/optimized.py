import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.cpp_extension import load

# Define the CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Helper function for hardswish activation
__device__ float hardswish(float x) {
    float temp = x + 3.0f;
    temp = (temp < 0.0f) ? 0.0f : ((temp > 6.0f) ? 6.0f : temp);
    return x * temp / 6.0f;
}

// Helper function for mish activation
__device__ float mish(float x) {
    return x * tanh(logf(1.0f + expf(x)));
}

// Main kernel for fused operations
__global__ void fused_conv_subtract_hardswish_maxpool_mish_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int height,
    const int width,
    const int kernel_size,
    const float subtract_value,
    const int pool_kernel_size,
    const int output_height,
    const int output_width) {
    
    // Calculate output indices
    const int n = blockIdx.z;
    const int f = blockIdx.y;
    const int y_out = (blockIdx.x / ((output_width + 7) / 8)) * blockDim.y + threadIdx.y;
    const int x_out = (blockIdx.x % ((output_width + 7) / 8)) * blockDim.x + threadIdx.x;
    
    // Check if within output bounds
    if (n >= batch_size || f >= out_channels || y_out >= output_height || x_out >= output_width)
        return;
    
    // Calculate input height and width after convolution but before pooling
    const int conv_height = height - kernel_size + 1;
    const int conv_width = width - kernel_size + 1;
    
    // Calculate the pooling region start
    const int pool_y_start = y_out * pool_kernel_size;
    const int pool_x_start = x_out * pool_kernel_size;
    
    // Variables for max pooling
    float max_val = -INFINITY;
    
    // Iterate over pooling region
    for (int py = 0; py < pool_kernel_size; py++) {
        for (int px = 0; px < pool_kernel_size; px++) {
            const int conv_y = pool_y_start + py;
            const int conv_x = pool_x_start + px;
            
            // Skip if outside conv output bounds
            if (conv_y >= conv_height || conv_x >= conv_width)
                continue;
            
            // Compute convolution for this position
            float conv_result = bias[f];
            
            // Iterate over input channels and kernel
            for (int c = 0; c < in_channels; c++) {
                for (int ky = 0; ky < kernel_size; ky++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        const int in_y = conv_y + ky;
                        const int in_x = conv_x + kx;
                        
                        // Get input and weight values
                        const float input_val = input[((n * in_channels + c) * height + in_y) * width + in_x];
                        const float weight_val = weight[((f * in_channels + c) * kernel_size + ky) * kernel_size + kx];
                        conv_result += input_val * weight_val;
                    }
                }
            }
            
            // Subtract value
            conv_result -= subtract_value;
            
            // Apply HardSwish
            float hardswish_val = hardswish(conv_result);
            
            // Update max value for pooling
            max_val = (hardswish_val > max_val) ? hardswish_val : max_val;
        }
    }
    
    // Apply Mish
    float mish_val = mish(max_val);
    
    // Write output
    output[((n * out_channels + f) * output_height + y_out) * output_width + x_out] = mish_val;
}

// Forward function that launches the CUDA kernel
torch::Tensor fused_conv_subtract_hardswish_maxpool_mish_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const float subtract_value,
    const int pool_kernel_size) {
    
    // Get dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    // Calculate output dimensions
    const int conv_height = height - kernel_size + 1;
    const int conv_width = width - kernel_size + 1;
    const int output_height = conv_height / pool_kernel_size;
    const int output_width = conv_width / pool_kernel_size;
    
    // Allocate output tensor
    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, 
                              input.options());
    
    // Define grid and block dimensions
    const dim3 threads(8, 8);
    const dim3 blocks(
        ((output_width + threads.x - 1) / threads.x) * ((output_height + threads.y - 1) / threads.y),
        out_channels,
        batch_size
    );
    
    // Launch the kernel
    fused_conv_subtract_hardswish_maxpool_mish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        subtract_value,
        pool_kernel_size,
        output_height,
        output_width
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_conv_subtract_hardswish_maxpool_mish_forward, "Fused Conv2d forward");
}
"""

# Create a temporary directory for the extension
import tempfile
import shutil
temp_dir = tempfile.mkdtemp()

try:
    # Write the CUDA code to a file
    with open(os.path.join(temp_dir, "fused_ops_cuda.cu"), "w") as f:
        f.write(cuda_source)
    
    # Load the extension
    fused_ops = load(
        name="fused_ops",
        sources=[os.path.join(temp_dir, "fused_ops_cuda.cu")],
        verbose=False
    )
except Exception as e:
    # Fallback if compilation fails
    fused_ops = None
    print(f"Failed to compile CUDA extension: {e}")
    print("Falling back to PyTorch implementation")

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, subtracts a value, applies HardSwish, 
    MaxPool, and Mish activation functions.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        subtract_value (float): Value to subtract after convolution
        pool_kernel_size (int): Size of the max pooling kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value = subtract_value
        self.pool_kernel_size = pool_kernel_size
        
        # Create weight and bias parameters directly
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # Initialize parameters using kaiming_uniform for weights and uniform for bias
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Flag to check if CUDA extension is available
        self.has_cuda_ext = fused_ops is not None
    
    def forward(self, x):
        # Use CUDA extension if available and input is on CUDA
        if self.has_cuda_ext and x.is_cuda:
            return fused_ops.forward(
                x, self.weight, self.bias, self.subtract_value, self.pool_kernel_size
            )
        else:
            # Fallback to PyTorch implementation
            x = F.conv2d(x, self.weight, self.bias)
            x = x - self.subtract_value
            x = F.hardswish(x)
            x = F.max_pool2d(x, self.pool_kernel_size)
            x = F.mish(x)
            return x

# Clean up the temporary directory
if 'temp_dir' in locals():
    try:
        shutil.rmtree(temp_dir)
    except:
        pass

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value = 0.5
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size]