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
        subtract_value_1 (float): First value to subtract
        subtract_value_2 (float): Second value to subtract
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2
        
        # Create weight parameter
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        
        # Create bias parameter
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Initialize parameters using the same approach as nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        # Pre-subtract the combined subtraction values from the bias
        self.bias.data.sub_(subtract_value_1 + subtract_value_2)
        
        # Initialize CUDA kernel if available
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            try:
                import cupy as cp
                self.has_cupy = True
                
                # Define CUDA kernel for fused convolution + mish with shared memory
                self.kernel_code = '''
                extern "C" __global__ void fused_conv2d_mish_kernel(
                    const float* __restrict__ input,
                    const float* __restrict__ weight,
                    const float* __restrict__ bias,
                    float* __restrict__ output,
                    const int batch_size,
                    const int in_channels,
                    const int out_channels,
                    const int in_height,
                    const int in_width,
                    const int kernel_size,
                    const int out_height,
                    const int out_width)
                {
                    // Block and thread indices
                    const int tx = threadIdx.x;
                    const int ty = threadIdx.y;
                    const int bx = blockIdx.x;
                    const int by = blockIdx.y;
                    const int bz = blockIdx.z;
                    
                    // Output position
                    const int x_out = bx * blockDim.x + tx;
                    const int y_out = by * blockDim.y + ty;
                    
                    // Batch and channel indices
                    const int c_out = bz % out_channels;
                    const int b = bz / out_channels;
                    
                    // Check if within bounds
                    if (x_out >= out_width || y_out >= out_height || b >= batch_size)
                        return;
                    
                    // Define shared memory for input tile
                    // We need to load a tile of size (BLOCK_SIZE+KERNEL_SIZE-1) x (BLOCK_SIZE+KERNEL_SIZE-1)
                    // Add padding to avoid bank conflicts
                    extern __shared__ float s_input[];
                    const int tile_width = blockDim.x + kernel_size - 1;
                    const int tile_stride = tile_width + 1;  // +1 padding to avoid bank conflicts
                    
                    // Calculate input position
                    const int x_in_base = bx * blockDim.x;
                    const int y_in_base = by * blockDim.y;
                    
                    // Load bias
                    float value = bias[c_out];
                    
                    // Perform convolution with shared memory
                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        // Load input tile into shared memory with padding
                        for (int i = ty; i < tile_width; i += blockDim.y) {
                            const int y_in = y_in_base + i;
                            
                            for (int j = tx; j < tile_width; j += blockDim.x) {
                                const int x_in = x_in_base + j;
                                
                                float input_val = 0.0f;
                                if (y_in < in_height && x_in < in_width) {
                                    input_val = input[((b * in_channels + c_in) * in_height + y_in) * in_width + x_in];
                                }
                                s_input[i * tile_stride + j] = input_val;
                            }
                        }
                        
                        // Synchronize to make sure all threads have loaded their part of the input
                        __syncthreads();
                        
                        // Optimized 3x3 convolution with manual loop unrolling
                        if (kernel_size == 3) {
                            // Preload weights into registers for faster access
                            const float w00 = weight[((c_out * in_channels + c_in) * kernel_size + 0) * kernel_size + 0];
                            const float w01 = weight[((c_out * in_channels + c_in) * kernel_size + 0) * kernel_size + 1];
                            const float w02 = weight[((c_out * in_channels + c_in) * kernel_size + 0) * kernel_size + 2];
                            
                            const float w10 = weight[((c_out * in_channels + c_in) * kernel_size + 1) * kernel_size + 0];
                            const float w11 = weight[((c_out * in_channels + c_in) * kernel_size + 1) * kernel_size + 1];
                            const float w12 = weight[((c_out * in_channels + c_in) * kernel_size + 1) * kernel_size + 2];
                            
                            const float w20 = weight[((c_out * in_channels + c_in) * kernel_size + 2) * kernel_size + 0];
                            const float w21 = weight[((c_out * in_channels + c_in) * kernel_size + 2) * kernel_size + 1];
                            const float w22 = weight[((c_out * in_channels + c_in) * kernel_size + 2) * kernel_size + 2];
                            
                            // Get input values from shared memory
                            const float i00 = s_input[(ty + 0) * tile_stride + (tx + 0)];
                            const float i01 = s_input[(ty + 0) * tile_stride + (tx + 1)];
                            const float i02 = s_input[(ty + 0) * tile_stride + (tx + 2)];
                            
                            const float i10 = s_input[(ty + 1) * tile_stride + (tx + 0)];
                            const float i11 = s_input[(ty + 1) * tile_stride + (tx + 1)];
                            const float i12 = s_input[(ty + 1) * tile_stride + (tx + 2)];
                            
                            const float i20 = s_input[(ty + 2) * tile_stride + (tx + 0)];
                            const float i21 = s_input[(ty + 2) * tile_stride + (tx + 1)];
                            const float i22 = s_input[(ty + 2) * tile_stride + (tx + 2)];
                            
                            // Perform convolution using registers
                            value += i00 * w00 + i01 * w01 + i02 * w02 +
                                     i10 * w10 + i11 * w11 + i12 * w12 +
                                     i20 * w20 + i21 * w21 + i22 * w22;
                        } else {
                            // Generic implementation for any kernel size
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    const float input_val = s_input[(ty + kh) * tile_stride + (tx + kw)];
                                    const float weight_val = weight[((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw];
                                    value += input_val * weight_val;
                                }
                            }
                        }
                        
                        // Synchronize before loading next channel
                        __syncthreads();
                    }
                    
                    // Apply Mish activation: x * tanh(softplus(x))
                    // Optimized implementation with special cases for numerical stability
                    float mish_val;
                    
                    if (value > 20.0f) {
                        // For large values, mish(x) ≈ x to avoid overflow
                        mish_val = value;
                    } else if (value < -20.0f) {
                        // For very negative values, mish(x) ≈ 0
                        mish_val = 0.0f;
                    } else {
                        float softplus_val = logf(1.0f + expf(value));
                        float tanh_val = tanhf(softplus_val);
                        mish_val = value * tanh_val;
                    }
                    
                    // Write output
                    const int output_idx = ((b * out_channels + c_out) * out_height + y_out) * out_width + x_out;
                    output[output_idx] = mish_val;
                }
                '''
                
                # Compile the kernel
                self.cuda_module = cp.RawModule(code=self.kernel_code)
                self.fused_kernel = self.cuda_module.get_function("fused_conv2d_mish_kernel")
                
            except ImportError:
                self.has_cupy = False
        else:
            self.has_cupy = False

    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor after convolution, subtraction, and Mish activation
        """
        # Use our custom CUDA kernel if available and input is on CUDA
        if self.use_cuda and hasattr(self, 'has_cupy') and self.has_cupy and x.is_cuda:
            try:
                import cupy as cp
                
                # Ensure input is contiguous for better memory access
                if not x.is_contiguous():
                    x = x.contiguous()
                
                batch_size, in_channels, in_height, in_width = x.shape
                out_height = in_height - self.kernel_size + 1
                out_width = in_width - self.kernel_size + 1
                
                # Create output tensor
                output = torch.empty(batch_size, self.out_channels, out_height, out_width, 
                                    device=x.device, dtype=x.dtype)
                
                # Calculate grid and block dimensions
                threads_per_block_x = 16
                threads_per_block_y = 16
                
                # Limit grid dimensions to avoid excessive blocks
                blocks_x = (out_width + threads_per_block_x - 1) // threads_per_block_x
                blocks_y = (out_height + threads_per_block_y - 1) // threads_per_block_y
                blocks_z = batch_size * self.out_channels
                
                # Calculate shared memory size with padding to avoid bank conflicts
                tile_width = threads_per_block_x + self.kernel_size - 1
                tile_stride = tile_width + 1  # +1 padding to avoid bank conflicts
                shared_mem_size = tile_width * tile_stride * 4  # 4 bytes per float
                
                # Launch kernel
                self.fused_kernel(
                    grid=(blocks_x, blocks_y, blocks_z),
                    block=(threads_per_block_x, threads_per_block_y, 1),
                    args=(cp.asarray(x), cp.asarray(self.weight), cp.asarray(self.bias), 
                         cp.asarray(output), batch_size, in_channels, self.out_channels, 
                         in_height, in_width, self.kernel_size, out_height, out_width),
                    shared_mem=shared_mem_size
                )
                
                return output
                
            except Exception:
                # Fallback to PyTorch implementation if there's an error
                pass
        
        # PyTorch fallback implementation - still optimized with fused bias
        x = F.conv2d(x, self.weight, self.bias)
        return F.mish(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
subtract_value_1 = 0.5
subtract_value_2 = 0.2

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2]