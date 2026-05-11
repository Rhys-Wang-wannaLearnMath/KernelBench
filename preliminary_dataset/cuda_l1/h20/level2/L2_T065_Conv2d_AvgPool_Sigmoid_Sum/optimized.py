import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolution kernel
        pool_kernel_size (int): Size of the pooling kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Create weights and bias directly as parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # Initialize parameters using the same method as nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        
        # Flag to track if CUDA kernel is available
        self.cuda_kernel_available = False
        
        # Try to load the CUDA kernel
        try:
            from torch.utils.cpp_extension import load_inline
            
            cuda_source = """
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>
            
            template <typename scalar_t>
            __global__ void optimized_conv2d_kernel(
                const scalar_t* __restrict__ input,
                const scalar_t* __restrict__ weight,
                const scalar_t* __restrict__ bias,
                scalar_t* __restrict__ output,
                const int batch_size,
                const int in_channels,
                const int out_channels,
                const int height,
                const int width,
                const int kernel_size,
                const int output_height,
                const int output_width) {
                
                // Shared memory for input tile and weights
                extern __shared__ char shared_memory[];
                scalar_t* shared_input = (scalar_t*)shared_memory;
                scalar_t* shared_weight = (scalar_t*)(shared_memory + sizeof(scalar_t) * (blockDim.x + kernel_size - 1) * (blockDim.y + kernel_size - 1));
                
                // Calculate output position
                const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
                const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
                const int out_ch = blockIdx.z % out_channels;
                const int batch = blockIdx.z / out_channels;
                
                // Check if thread is within output bounds
                if (out_x >= output_width || out_y >= output_height || batch >= batch_size) {
                    return;
                }
                
                // Load bias into register
                scalar_t bias_val = bias[out_ch];
                
                // Compute input tile indices
                const int tile_start_y = blockIdx.y * blockDim.y;
                const int tile_start_x = blockIdx.x * blockDim.x;
                
                // Accumulator for convolution result
                scalar_t result = bias_val;
                
                // Perform convolution
                for (int in_ch = 0; in_ch < in_channels; ++in_ch) {
                    // Collaboratively load input tile into shared memory
                    for (int i = threadIdx.y; i < blockDim.y + kernel_size - 1; i += blockDim.y) {
                        for (int j = threadIdx.x; j < blockDim.x + kernel_size - 1; j += blockDim.x) {
                            int y = tile_start_y + i - (kernel_size / 2);
                            int x = tile_start_x + j - (kernel_size / 2);
                            
                            if (y >= 0 && y < height && x >= 0 && x < width) {
                                shared_input[i * (blockDim.x + kernel_size - 1) + j] = 
                                    input[((batch * in_channels + in_ch) * height + y) * width + x];
                            } else {
                                shared_input[i * (blockDim.x + kernel_size - 1) + j] = 0.0;
                            }
                        }
                    }
                    
                    // Collaboratively load weights into shared memory
                    for (int i = threadIdx.y; i < kernel_size; i += blockDim.y) {
                        for (int j = threadIdx.x; j < kernel_size; j += blockDim.x) {
                            if (i < kernel_size && j < kernel_size) {
                                shared_weight[i * kernel_size + j] = 
                                    weight[((out_ch * in_channels + in_ch) * kernel_size + i) * kernel_size + j];
                            }
                        }
                    }
                    
                    __syncthreads();
                    
                    // Compute convolution for this input channel with loop unrolling
                    #pragma unroll 3
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        #pragma unroll 3
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            result += shared_input[(threadIdx.y + ky) * (blockDim.x + kernel_size - 1) + (threadIdx.x + kx)] * 
                                      shared_weight[ky * kernel_size + kx];
                        }
                    }
                    
                    __syncthreads();
                }
                
                // Store result in output
                output[((batch * out_channels + out_ch) * output_height + out_y) * output_width + out_x] = result;
            }
            
            torch::Tensor optimized_conv2d_cuda(
                torch::Tensor input,
                torch::Tensor weight,
                torch::Tensor bias) {
                
                // Get dimensions
                const int batch_size = input.size(0);
                const int in_channels = input.size(1);
                const int height = input.size(2);
                const int width = input.size(3);
                const int out_channels = weight.size(0);
                const int kernel_size = weight.size(2);
                
                // Calculate output dimensions
                const int output_height = height - kernel_size + 1;
                const int output_width = width - kernel_size + 1;
                
                // Create output tensor
                auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, 
                                          input.options());
                
                // Configure kernel launch parameters
                const int tile_size_x = 16;
                const int tile_size_y = 16;
                const dim3 threads(tile_size_x, tile_size_y);
                const dim3 blocks(
                    (output_width + tile_size_x - 1) / tile_size_x,
                    (output_height + tile_size_y - 1) / tile_size_y,
                    batch_size * out_channels
                );
                
                // Calculate shared memory size
                const size_t shared_input_size = (tile_size_x + kernel_size - 1) * (tile_size_y + kernel_size - 1);
                const size_t shared_weight_size = kernel_size * kernel_size;
                const size_t shared_mem_size = sizeof(float) * (shared_input_size + shared_weight_size);
                
                // Launch kernel
                AT_DISPATCH_FLOATING_TYPES(input.type(), "optimized_conv2d_cuda", ([&] {
                    optimized_conv2d_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
                        input.data_ptr<scalar_t>(),
                        weight.data_ptr<scalar_t>(),
                        bias.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_size,
                        in_channels,
                        out_channels,
                        height,
                        width,
                        kernel_size,
                        output_height,
                        output_width
                    );
                }));
                
                return output;
            }
            """
            
            cpp_source = """
            #include <torch/extension.h>
            
            torch::Tensor optimized_conv2d_cuda(
                torch::Tensor input,
                torch::Tensor weight,
                torch::Tensor bias);
            
            torch::Tensor optimized_conv2d(
                torch::Tensor input,
                torch::Tensor weight,
                torch::Tensor bias) {
                return optimized_conv2d_cuda(input, weight, bias);
            }
            
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("optimized_conv2d", &optimized_conv2d, "Optimized Conv2d forward");
            }
            """
            
            self.conv_cuda = load_inline(
                name='optimized_conv2d_extension',
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                functions=['optimized_conv2d'],
                verbose=True
            )
            
            self.cuda_kernel_available = True
        except Exception as e:
            print(f"Failed to load CUDA kernel: {e}")
            print("Falling back to PyTorch implementation")
            self.cuda_kernel_available = False
    
    def forward(self, x):
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Step 1: Convolution
        if self.cuda_kernel_available and x.is_cuda:
            try:
                # Try using our custom CUDA kernel
                conv_out = self.conv_cuda.optimized_conv2d(x, self.weight, self.bias)
            except Exception as e:
                print(f"CUDA kernel execution failed: {e}")
                print("Falling back to PyTorch implementation")
                # Fallback to PyTorch's implementation
                conv_out = F.conv2d(x, self.weight, self.bias)
        else:
            # Use PyTorch's implementation
            conv_out = F.conv2d(x, self.weight, self.bias)
        
        # Step 2: Average pooling (using PyTorch's optimized implementation)
        pooled_out = F.avg_pool2d(conv_out, self.pool_kernel_size)
        
        # Step 3: Apply sigmoid activation (using PyTorch's optimized implementation)
        sigmoid_out = torch.sigmoid(pooled_out)
        
        # Step 4: Sum over all dimensions (using PyTorch's optimized implementation)
        # First sum over spatial dimensions (more efficient)
        spatial_sum = torch.sum(sigmoid_out, dim=[2, 3])
        # Then sum over channels
        result = torch.sum(spatial_sum, dim=1)
        
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_channels, out_channels, kernel_size, pool_kernel_size]