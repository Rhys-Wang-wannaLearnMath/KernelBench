import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline
import os

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    This implementation uses an optimized custom CUDA kernel for better performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # CUDA kernel source code
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <vector>

        // Constants for the kernel
        #define BLOCK_SIZE_X 8
        #define BLOCK_SIZE_Y 8
        #define WARP_SIZE 32

        // Fast math implementation of mish activation
        template <typename scalar_t>
        __device__ __forceinline__ scalar_t mish_activation(scalar_t x) {
            // Optimized Mish implementation: x * tanh(softplus(x))
            if (x <= -20.0f) {
                // For very negative inputs, avoid overflow
                return 0.0f;
            } else if (x >= 20.0f) {
                // For very positive inputs, mish(x) ≈ x
                return x;
            } else {
                scalar_t sp = log1pf(expf(x));
                return x * tanhf(sp);
            }
        }

        template <typename scalar_t>
        __global__ void conv3d_mish_tanh_kernel(
            const scalar_t* __restrict__ input,
            const scalar_t* __restrict__ weight,
            const scalar_t* __restrict__ bias,
            scalar_t* __restrict__ output,
            int batch_size, int in_channels, int out_channels,
            int depth, int height, int width,
            int out_depth, int out_height, int out_width,
            int kernel_size, int stride, int padding) {
            
            // Shared memory for weights
            extern __shared__ char shared_memory[];
            scalar_t* shared_weights = (scalar_t*)shared_memory;
            
            // Thread indices
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int thread_id = ty * BLOCK_SIZE_X + tx;
            
            // Block indices
            const int bx = blockIdx.x;
            const int by = blockIdx.y;
            const int bz = blockIdx.z;
            
            // Calculate batch and channel indices
            const int batch_out_channel = bz;
            const int out_c = batch_out_channel % out_channels;
            const int batch_idx = batch_out_channel / out_channels;
            
            if (batch_idx >= batch_size) return;
            
            // Load weights into shared memory - collaborative loading
            const int weight_elements = in_channels * kernel_size * kernel_size * kernel_size;
            const int threads_per_block = BLOCK_SIZE_X * BLOCK_SIZE_Y;
            
            for (int i = thread_id; i < weight_elements; i += threads_per_block) {
                const int ic = i / (kernel_size * kernel_size * kernel_size);
                const int remainder = i % (kernel_size * kernel_size * kernel_size);
                const int kz = remainder / (kernel_size * kernel_size);
                const int remainder2 = remainder % (kernel_size * kernel_size);
                const int ky = remainder2 / kernel_size;
                const int kx = remainder2 % kernel_size;
                
                shared_weights[i] = weight[((out_c * in_channels + ic) * kernel_size + kz) * 
                                          kernel_size * kernel_size + ky * kernel_size + kx];
            }
            __syncthreads();
            
            // Calculate output positions using grid-stride loop
            for (int out_z = 0; out_z < out_depth; out_z++) {
                for (int out_y_base = by * BLOCK_SIZE_Y; out_y_base < out_height; out_y_base += gridDim.y * BLOCK_SIZE_Y) {
                    const int out_y = out_y_base + ty;
                    
                    if (out_y >= out_height) continue;
                    
                    for (int out_x_base = bx * BLOCK_SIZE_X; out_x_base < out_width; out_x_base += gridDim.x * BLOCK_SIZE_X) {
                        const int out_x = out_x_base + tx;
                        
                        if (out_x >= out_width) continue;
                        
                        // Load bias
                        scalar_t result = bias[out_c];
                        
                        // Compute convolution
                        #pragma unroll 3
                        for (int ic = 0; ic < in_channels; ic++) {
                            #pragma unroll 3
                            for (int kz = 0; kz < kernel_size; kz++) {
                                const int in_z = out_z * stride - padding + kz;
                                
                                if (in_z >= 0 && in_z < depth) {
                                    #pragma unroll 3
                                    for (int ky = 0; ky < kernel_size; ky++) {
                                        const int in_y = out_y * stride - padding + ky;
                                        
                                        if (in_y >= 0 && in_y < height) {
                                            #pragma unroll 3
                                            for (int kx = 0; kx < kernel_size; kx++) {
                                                const int in_x = out_x * stride - padding + kx;
                                                
                                                if (in_x >= 0 && in_x < width) {
                                                    const int input_idx = ((batch_idx * in_channels + ic) * depth + in_z) * 
                                                                        height * width + in_y * width + in_x;
                                                    const int weight_idx = (ic * kernel_size + kz) * kernel_size * kernel_size + 
                                                                         ky * kernel_size + kx;
                                                    
                                                    result += input[input_idx] * shared_weights[weight_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Apply Mish activation
                        result = mish_activation(result);
                        
                        // Apply Tanh activation
                        result = tanhf(result);
                        
                        // Write output
                        const int output_idx = ((batch_idx * out_channels + out_c) * out_depth + out_z) * 
                                              out_height * out_width + out_y * out_width + out_x;
                        output[output_idx] = result;
                    }
                }
            }
        }

        // Kernel for vectorized processing using float4
        template <typename scalar_t>
        __global__ void conv3d_mish_tanh_vec4_kernel(
            const scalar_t* __restrict__ input,
            const scalar_t* __restrict__ weight,
            const scalar_t* __restrict__ bias,
            scalar_t* __restrict__ output,
            int batch_size, int in_channels, int out_channels,
            int depth, int height, int width,
            int out_depth, int out_height, int out_width,
            int kernel_size, int stride, int padding) {
            
            // Shared memory for weights
            extern __shared__ char shared_memory[];
            scalar_t* shared_weights = (scalar_t*)shared_memory;
            
            // Thread indices
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            const int thread_id = ty * BLOCK_SIZE_X + tx;
            
            // Block indices
            const int bx = blockIdx.x;
            const int by = blockIdx.y;
            const int bz = blockIdx.z;
            
            // Calculate batch and channel indices
            const int batch_out_channel = bz;
            const int out_c = batch_out_channel % out_channels;
            const int batch_idx = batch_out_channel / out_channels;
            
            if (batch_idx >= batch_size) return;
            
            // Load weights into shared memory - collaborative loading
            const int weight_elements = in_channels * kernel_size * kernel_size * kernel_size;
            const int threads_per_block = BLOCK_SIZE_X * BLOCK_SIZE_Y;
            
            for (int i = thread_id; i < weight_elements; i += threads_per_block) {
                const int ic = i / (kernel_size * kernel_size * kernel_size);
                const int remainder = i % (kernel_size * kernel_size * kernel_size);
                const int kz = remainder / (kernel_size * kernel_size);
                const int remainder2 = remainder % (kernel_size * kernel_size);
                const int ky = remainder2 / kernel_size;
                const int kx = remainder2 % kernel_size;
                
                shared_weights[i] = weight[((out_c * in_channels + ic) * kernel_size + kz) * 
                                          kernel_size * kernel_size + ky * kernel_size + kx];
            }
            __syncthreads();
            
            // Process 4 output elements at once along x dimension when possible
            const int out_x_base = bx * BLOCK_SIZE_X * 4 + tx * 4;
            const int out_y = by * BLOCK_SIZE_Y + ty;
            
            if (out_y < out_height) {
                // Process multiple depth planes per thread block
                for (int out_z = 0; out_z < out_depth; out_z++) {
                    // Check if we can process 4 elements at once
                    if (out_x_base < out_width - 3) {
                        // Pre-load bias for 4 elements
                        scalar_t result0 = bias[out_c];
                        scalar_t result1 = bias[out_c];
                        scalar_t result2 = bias[out_c];
                        scalar_t result3 = bias[out_c];
                        
                        // Compute convolution for 4 output elements
                        #pragma unroll 3
                        for (int ic = 0; ic < in_channels; ic++) {
                            #pragma unroll 3
                            for (int kz = 0; kz < kernel_size; kz++) {
                                const int in_z = out_z * stride - padding + kz;
                                
                                if (in_z >= 0 && in_z < depth) {
                                    #pragma unroll 3
                                    for (int ky = 0; ky < kernel_size; ky++) {
                                        const int in_y = out_y * stride - padding + ky;
                                        
                                        if (in_y >= 0 && in_y < height) {
                                            #pragma unroll 3
                                            for (int kx = 0; kx < kernel_size; kx++) {
                                                const int weight_idx = (ic * kernel_size + kz) * kernel_size * kernel_size + 
                                                                     ky * kernel_size + kx;
                                                const scalar_t w = shared_weights[weight_idx];
                                                
                                                // Process 4 output elements
                                                for (int i = 0; i < 4; i++) {
                                                    const int out_x = out_x_base + i;
                                                    const int in_x = out_x * stride - padding + kx;
                                                    
                                                    if (in_x >= 0 && in_x < width) {
                                                        const int input_idx = ((batch_idx * in_channels + ic) * depth + in_z) * 
                                                                            height * width + in_y * width + in_x;
                                                        
                                                        // Add to appropriate result
                                                        if (i == 0) result0 += input[input_idx] * w;
                                                        else if (i == 1) result1 += input[input_idx] * w;
                                                        else if (i == 2) result2 += input[input_idx] * w;
                                                        else result3 += input[input_idx] * w;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Apply Mish and Tanh activations
                        result0 = tanhf(mish_activation(result0));
                        result1 = tanhf(mish_activation(result1));
                        result2 = tanhf(mish_activation(result2));
                        result3 = tanhf(mish_activation(result3));
                        
                        // Write output
                        const int base_output_idx = ((batch_idx * out_channels + out_c) * out_depth + out_z) * 
                                                  out_height * out_width + out_y * out_width;
                        
                        output[base_output_idx + out_x_base] = result0;
                        output[base_output_idx + out_x_base + 1] = result1;
                        output[base_output_idx + out_x_base + 2] = result2;
                        output[base_output_idx + out_x_base + 3] = result3;
                    }
                    else {
                        // Handle boundary case with individual processing
                        for (int i = 0; i < 4; i++) {
                            const int out_x = out_x_base + i;
                            if (out_x < out_width) {
                                // Load bias
                                scalar_t result = bias[out_c];
                                
                                // Compute convolution
                                #pragma unroll 3
                                for (int ic = 0; ic < in_channels; ic++) {
                                    #pragma unroll 3
                                    for (int kz = 0; kz < kernel_size; kz++) {
                                        const int in_z = out_z * stride - padding + kz;
                                        
                                        if (in_z >= 0 && in_z < depth) {
                                            #pragma unroll 3
                                            for (int ky = 0; ky < kernel_size; ky++) {
                                                const int in_y = out_y * stride - padding + ky;
                                                
                                                if (in_y >= 0 && in_y < height) {
                                                    #pragma unroll 3
                                                    for (int kx = 0; kx < kernel_size; kx++) {
                                                        const int in_x = out_x * stride - padding + kx;
                                                        
                                                        if (in_x >= 0 && in_x < width) {
                                                            const int input_idx = ((batch_idx * in_channels + ic) * depth + in_z) * 
                                                                                height * width + in_y * width + in_x;
                                                            const int weight_idx = (ic * kernel_size + kz) * kernel_size * kernel_size + 
                                                                                 ky * kernel_size + kx;
                                                            
                                                            result += input[input_idx] * shared_weights[weight_idx];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                // Apply Mish activation
                                result = mish_activation(result);
                                
                                // Apply Tanh activation
                                result = tanhf(result);
                                
                                // Write output
                                const int output_idx = ((batch_idx * out_channels + out_c) * out_depth + out_z) * 
                                                      out_height * out_width + out_y * out_width + out_x;
                                output[output_idx] = result;
                            }
                        }
                    }
                }
            }
        }

        torch::Tensor conv3d_mish_tanh_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int stride,
            int padding) {
            
            // Get dimensions
            const int batch_size = input.size(0);
            const int in_channels = input.size(1);
            const int depth = input.size(2);
            const int height = input.size(3);
            const int width = input.size(4);
            
            const int out_channels = weight.size(0);
            const int kernel_size = weight.size(2);
            
            // Calculate output dimensions
            const int out_depth = (depth + 2 * padding - kernel_size) / stride + 1;
            const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
            const int out_width = (width + 2 * padding - kernel_size) / stride + 1;
            
            // Create output tensor
            auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, 
                                      input.options());
            
            // Configure kernel
            const dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            
            // Calculate shared memory size for weights
            const int shared_mem_size = in_channels * kernel_size * kernel_size * kernel_size * sizeof(float);
            
            // Choose kernel based on output width
            if (out_width >= 32) {
                // Use vectorized kernel for larger widths
                // Adjust grid dimensions for vectorized processing (each thread handles 4 output elements)
                const int grid_x = (out_width + BLOCK_SIZE_X * 4 - 1) / (BLOCK_SIZE_X * 4);
                const int grid_y = (out_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
                const int grid_z = batch_size * out_channels;
                
                const dim3 blocks(grid_x, grid_y, grid_z);
                
                AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_mish_tanh_vec4_cuda", ([&] {
                    conv3d_mish_tanh_vec4_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
                        input.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(),
                        bias.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_size, in_channels, out_channels,
                        depth, height, width,
                        out_depth, out_height, out_width,
                        kernel_size, stride, padding);
                }));
            } else {
                // Use standard kernel for smaller widths
                const int grid_x = std::min(32, (out_width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X);
                const int grid_y = std::min(32, (out_height + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
                const int grid_z = batch_size * out_channels;
                
                const dim3 blocks(grid_x, grid_y, grid_z);
                
                AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_mish_tanh_cuda", ([&] {
                    conv3d_mish_tanh_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
                        input.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(),
                        bias.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_size, in_channels, out_channels,
                        depth, height, width,
                        out_depth, out_height, out_width,
                        kernel_size, stride, padding);
                }));
            }
            
            return output;
        }
        """

        cpp_source = """
        #include <torch/extension.h>

        torch::Tensor conv3d_mish_tanh_cuda(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int stride,
            int padding);

        torch::Tensor conv3d_mish_tanh(
            torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor bias,
            int stride,
            int padding) {
            return conv3d_mish_tanh_cuda(input, weight, bias, stride, padding);
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("conv3d_mish_tanh", &conv3d_mish_tanh, "Conv3d with Mish and Tanh activation");
        }
        """
        
        # Compile the CUDA kernel
        if torch.cuda.is_available():
            # Create a unique name for the extension to avoid conflicts
            extension_name = f"conv3d_mish_tanh_opt_{os.getpid()}"
            
            # Load the custom CUDA kernel
            try:
                self.custom_ops = load_inline(
                    name=extension_name,
                    cpp_sources=cpp_source,
                    cuda_sources=cuda_source,
                    functions=["conv3d_mish_tanh"],
                    verbose=False,
                    with_cuda=True,
                    extra_cuda_cflags=['-O3', '--use_fast_math']
                )
                self.use_custom_kernel = True
            except Exception as e:
                print(f"Failed to compile CUDA kernel: {e}")
                self.use_custom_kernel = False
        else:
            self.use_custom_kernel = False

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        if self.use_custom_kernel and x.is_cuda:
            # Use the custom CUDA kernel for the entire operation
            return self.custom_ops.conv3d_mish_tanh(
                x, self.weight, self.bias, self.stride, self.padding
            )
        else:
            # Fallback to PyTorch's built-in operations
            x = F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
            x = F.mish(x)
            x = torch.tanh(x)
            return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]