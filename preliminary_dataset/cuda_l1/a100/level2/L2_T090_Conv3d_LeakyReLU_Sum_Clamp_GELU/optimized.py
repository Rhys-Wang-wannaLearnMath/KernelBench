import torch
import torch.nn as nn
import torch.utils.cpp_extension
import os
import math

# Enable cuDNN benchmark mode to find the best algorithm
torch.backends.cudnn.benchmark = True

# Create a directory for our CUDA code
os.makedirs('cuda_code', exist_ok=True)

# Write CUDA kernel code
with open('cuda_code/fused_ops_kernel.cu', 'w') as f:
    f.write('''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel for fused operations: LeakyReLU -> Add -> Clamp -> GELU
__global__ void fused_ops_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ sum_tensor,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    float negative_slope) {
    
    const int total_elements = batch_size * channels * depth * height * width;
    const int dhw = depth * height * width;
    
    // Use vectorized loads where possible
    const int vector_size = 4;  // float4
    const int vector_elements = total_elements / vector_size;
    
    // Process elements in chunks of 4 where possible
    for (int base_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         base_idx < vector_elements; 
         base_idx += blockDim.x * gridDim.x) {
        
        int idx = base_idx * vector_size;
        
        // Process 4 elements at once
        #pragma unroll
        for (int i = 0; i < vector_size; i++) {
            const int curr_idx = idx + i;
            
            // Calculate channel index for broadcasting
            const int c = (curr_idx / dhw) % channels;
            
            // Get input value
            const float x = input[curr_idx];
            
            // Apply LeakyReLU
            float result = x > 0 ? x : x * negative_slope;
            
            // Add sum_tensor (broadcasting)
            result += sum_tensor[c];
            
            // Apply clamp
            result = fminf(1.0f, fmaxf(-1.0f, result));
            
            // Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const float sqrt_2_pi_inv = 0.7978845608028654f;  // sqrt(2/pi)
            const float coeff = 0.044715f;
            const float x_cubed = result * result * result;
            const float inner = sqrt_2_pi_inv * (result + coeff * x_cubed);
            result = 0.5f * result * (1.0f + tanhf(inner));
            
            // Store result
            output[curr_idx] = result;
        }
    }
    
    // Handle remaining elements
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x + vector_elements * vector_size; 
         idx < total_elements; 
         idx += blockDim.x * gridDim.x) {
        
        // Calculate channel index for broadcasting
        const int c = (idx / dhw) % channels;
        
        // Get input value
        const float x = input[idx];
        
        // Apply LeakyReLU
        float result = x > 0 ? x : x * negative_slope;
        
        // Add sum_tensor (broadcasting)
        result += sum_tensor[c];
        
        // Apply clamp
        result = fminf(1.0f, fmaxf(-1.0f, result));
        
        // Fast GELU approximation
        const float sqrt_2_pi_inv = 0.7978845608028654f;
        const float coeff = 0.044715f;
        const float x_cubed = result * result * result;
        const float inner = sqrt_2_pi_inv * (result + coeff * x_cubed);
        result = 0.5f * result * (1.0f + tanhf(inner));
        
        // Store result
        output[idx] = result;
    }
}

torch::Tensor fused_ops_forward_cuda(
    torch::Tensor input,
    torch::Tensor sum_tensor,
    float negative_slope) {
    
    auto output = torch::zeros_like(input);
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    // Optimize thread and block configuration
    const int threads = 256;
    const int blocks = min(65535, (batch_size * channels * depth * height * width + threads - 1) / threads);
    
    fused_ops_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        negative_slope
    );
    
    return output;
}
''')

# Write C++ interface
with open('cuda_code/fused_ops.cpp', 'w') as f:
    f.write('''
#include <torch/extension.h>

// Forward declaration of CUDA function
torch::Tensor fused_ops_forward_cuda(
    torch::Tensor input,
    torch::Tensor sum_tensor,
    float negative_slope);

// C++ interface
torch::Tensor fused_ops_forward(
    torch::Tensor input,
    torch::Tensor sum_tensor,
    float negative_slope) {
    
    return fused_ops_forward_cuda(input, sum_tensor, negative_slope);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_ops_forward, "Fused Ops Forward");
}
''')

# Try to compile the extension
try:
    fused_ops_cuda = torch.utils.cpp_extension.load(
        name="fused_ops_cuda",
        sources=["cuda_code/fused_ops.cpp", "cuda_code/fused_ops_kernel.cu"],
        verbose=True,
        extra_cuda_cflags=["--use_fast_math", "-O3"]  # Enable fast math and high optimization
    )
    has_cuda_extension = True
except Exception as e:
    print(f"Failed to compile CUDA extension: {e}")
    print("Falling back to PyTorch implementation.")
    has_cuda_extension = False

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.has_cuda_extension = has_cuda_extension
        
        # Convert weights to channels_last_3d format during initialization
        self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last_3d)
        if self.conv.bias is not None:
            self.conv.bias.data = self.conv.bias.data.contiguous()

    def forward(self, x):
        # Convert input to channels_last_3d format for better memory access patterns
        x = x.to(memory_format=torch.channels_last_3d)
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply fused element-wise operations
        if self.has_cuda_extension and x.is_cuda and self.sum_tensor.is_cuda:
            try:
                # Use custom CUDA kernel if available and tensors are on GPU
                return fused_ops_cuda.forward(x, self.sum_tensor, 0.2)
            except Exception as e:
                print(f"Error in CUDA kernel: {e}")
                print("Falling back to PyTorch implementation.")
        
        # Fallback to PyTorch implementation
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = x + self.sum_tensor
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = torch.nn.functional.gelu(x)
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]