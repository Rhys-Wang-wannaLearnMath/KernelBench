import torch
import torch.nn as nn
import torch.utils.cpp_extension
import os

class ModelNew(nn.Module):
    """
    Optimized model that performs a LogSoftmax activation.
    """
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim
        
        # Define CUDA kernel for LogSoftmax
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <vector>

        // Fast CUDA warp reduction for maximum value using shuffle xor
        template <typename scalar_t>
        __device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
            return val;
        }

        // Fast CUDA warp reduction for sum using shuffle xor
        template <typename scalar_t>
        __device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                val += __shfl_xor_sync(0xffffffff, val, offset);
            return val;
        }

        // Optimized CUDA kernel for LogSoftmax forward pass
        template <typename scalar_t>
        __global__ void log_softmax_forward_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int dim) {
            
            // Each block handles one row (one sample in the batch)
            const int batch_idx = blockIdx.x;
            if (batch_idx >= batch_size) return;
            
            // Get pointers to current row
            const scalar_t* row_input = input + batch_idx * dim;
            scalar_t* row_output = output + batch_idx * dim;
            
            // Shared memory for reductions
            extern __shared__ char shared_mem[];
            scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
            
            const int tid = threadIdx.x;
            const int lane_id = tid % 32;
            const int warp_id = tid / 32;
            const int warps_per_block = blockDim.x / 32;
            
            // Step 1: Find max value in this row
            scalar_t thread_max = -INFINITY;
            
            // Each thread processes multiple elements with stride for better memory coalescing
            // Process elements in chunks of 4 when possible for better memory throughput
            if (sizeof(scalar_t) == sizeof(float) && dim % 4 == 0) {
                const float4* row_input4 = reinterpret_cast<const float4*>(row_input);
                const int vec_dim = dim / 4;
                
                #pragma unroll 4
                for (int i = tid; i < vec_dim; i += blockDim.x) {
                    float4 vec = row_input4[i];
                    thread_max = max(thread_max, static_cast<scalar_t>(vec.x));
                    thread_max = max(thread_max, static_cast<scalar_t>(vec.y));
                    thread_max = max(thread_max, static_cast<scalar_t>(vec.z));
                    thread_max = max(thread_max, static_cast<scalar_t>(vec.w));
                }
            } else {
                #pragma unroll 8
                for (int i = tid; i < dim; i += blockDim.x) {
                    thread_max = max(thread_max, row_input[i]);
                }
            }
            
            // Warp-level reduction for max
            thread_max = warp_reduce_max(thread_max);
            
            // Store the warp-level results
            if (lane_id == 0) {
                shared_data[warp_id] = thread_max;
            }
            __syncthreads();
            
            // Final reduction for max across warps
            if (warp_id == 0) {
                scalar_t warp_max = -INFINITY;
                if (tid < warps_per_block) {
                    warp_max = shared_data[tid];
                }
                warp_max = warp_reduce_max(warp_max);
                
                if (lane_id == 0) {
                    shared_data[0] = warp_max;
                }
            }
            __syncthreads();
            
            // Get the block-wide maximum
            const scalar_t row_max = shared_data[0];
            
            // Step 2: Compute sum of exp(x - max)
            scalar_t thread_sum = 0;
            
            // Each thread processes multiple elements with stride for better memory coalescing
            // Process elements in chunks of 4 when possible for better memory throughput
            if (sizeof(scalar_t) == sizeof(float) && dim % 4 == 0) {
                const float4* row_input4 = reinterpret_cast<const float4*>(row_input);
                const int vec_dim = dim / 4;
                
                #pragma unroll 4
                for (int i = tid; i < vec_dim; i += blockDim.x) {
                    float4 vec = row_input4[i];
                    thread_sum += exp(static_cast<scalar_t>(vec.x) - row_max);
                    thread_sum += exp(static_cast<scalar_t>(vec.y) - row_max);
                    thread_sum += exp(static_cast<scalar_t>(vec.z) - row_max);
                    thread_sum += exp(static_cast<scalar_t>(vec.w) - row_max);
                }
            } else {
                #pragma unroll 8
                for (int i = tid; i < dim; i += blockDim.x) {
                    thread_sum += exp(row_input[i] - row_max);
                }
            }
            
            // Warp-level reduction for sum
            thread_sum = warp_reduce_sum(thread_sum);
            
            // Store the warp-level results
            if (lane_id == 0) {
                shared_data[warp_id] = thread_sum;
            }
            __syncthreads();
            
            // Final reduction for sum across warps
            if (warp_id == 0) {
                scalar_t warp_sum = 0;
                if (tid < warps_per_block) {
                    warp_sum = shared_data[tid];
                }
                warp_sum = warp_reduce_sum(warp_sum);
                
                if (lane_id == 0) {
                    shared_data[0] = warp_sum;
                }
            }
            __syncthreads();
            
            // Get the block-wide sum and compute log
            const scalar_t sum = shared_data[0];
            const scalar_t log_sum = log(sum);
            
            // Step 3: Compute final output: x - max - log(sum(exp(x - max)))
            // Each thread processes multiple elements with stride for better memory coalescing
            // Process elements in chunks of 4 when possible for better memory throughput
            if (sizeof(scalar_t) == sizeof(float) && dim % 4 == 0) {
                const float4* row_input4 = reinterpret_cast<const float4*>(row_input);
                float4* row_output4 = reinterpret_cast<float4*>(row_output);
                const int vec_dim = dim / 4;
                
                #pragma unroll 4
                for (int i = tid; i < vec_dim; i += blockDim.x) {
                    float4 vec_in = row_input4[i];
                    float4 vec_out;
                    
                    vec_out.x = vec_in.x - row_max - log_sum;
                    vec_out.y = vec_in.y - row_max - log_sum;
                    vec_out.z = vec_in.z - row_max - log_sum;
                    vec_out.w = vec_in.w - row_max - log_sum;
                    
                    row_output4[i] = vec_out;
                }
            } else {
                #pragma unroll 8
                for (int i = tid; i < dim; i += blockDim.x) {
                    row_output[i] = row_input[i] - row_max - log_sum;
                }
            }
        }

        // C++ interface for the CUDA kernels
        std::vector<torch::Tensor> log_softmax_cuda_forward(
            torch::Tensor input,
            int dim) {
            
            // Check input
            TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
            TORCH_CHECK(dim >= 0 && dim < input.dim(), "Invalid dimension");
            TORCH_CHECK(dim == 1, "Custom CUDA kernel only supports dim=1");
            TORCH_CHECK(input.dim() == 2, "Custom CUDA kernel only supports 2D tensors");
            
            // Get tensor dimensions
            const int batch_size = input.size(0);
            const int feature_dim = input.size(1);
            
            // Create output tensor
            auto output = torch::empty_like(input);
            
            // Get pointers to data
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax_forward_cuda", ([&] {
                const scalar_t* input_ptr = input.data_ptr<scalar_t>();
                scalar_t* output_ptr = output.data_ptr<scalar_t>();
                
                // Calculate thread count and shared memory size
                const int threads = 128;  // 128 threads per block showed best performance
                const int warps_per_block = threads / 32;
                const size_t shared_mem_size = sizeof(scalar_t) * warps_per_block;
                
                // Launch kernel
                log_softmax_forward_kernel<scalar_t><<<batch_size, threads, shared_mem_size>>>(
                    input_ptr, output_ptr, batch_size, feature_dim);
            }));
            
            return {output};
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &log_softmax_cuda_forward, "LogSoftmax forward (CUDA)");
        }
        """
        
        # Try to load the extension, with fallback to PyTorch implementation
        self.cuda_extension_available = False
        if torch.cuda.is_available():
            try:
                # Create a directory for storing compiled extensions if it doesn't exist
                os.makedirs('cuda_extensions', exist_ok=True)
                
                # Load the CUDA extension
                self.log_softmax_cuda = torch.utils.cpp_extension.load_inline(
                    name='log_softmax_cuda',
                    cpp_sources='',
                    cuda_sources=cuda_source,
                    functions=['forward'],
                    with_cuda=True,
                    build_directory='cuda_extensions',
                    verbose=False,
                    extra_cuda_cflags=['-O3', '--use_fast_math']
                )
                
                self.cuda_extension_available = True
            except Exception as e:
                print(f"Warning: Could not load CUDA extension: {e}")
                print("Falling back to PyTorch implementation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        # Use our custom CUDA implementation if available and applicable
        if self.cuda_extension_available and x.is_cuda and self.dim == 1 and x.dim() == 2:
            try:
                # Make sure input is contiguous
                if not x.is_contiguous():
                    x = x.contiguous()
                
                # Forward pass through custom CUDA kernel
                return self.log_softmax_cuda.forward(x, self.dim)[0]
            except Exception as e:
                # Fall back to PyTorch implementation if there's an error
                print(f"Warning: Custom CUDA kernel failed with error: {e}")
                print("Falling back to PyTorch implementation")
                return torch.log_softmax(x, self.dim)
        else:
            # Fall back to PyTorch implementation
            return torch.log_softmax(x, self.dim)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, dim, device=device)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed