import torch
import torch.nn as nn
import torch.utils.cpp_extension

class ModelNew(nn.Module):
    """
    Optimized model that performs a SELU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        # SELU constants
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        
        # CUDA kernel for SELU activation
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <vector_types.h>
        
        __constant__ float ALPHA = 1.6732632423543772848170429916717f;
        __constant__ float SCALE = 1.0507009873554804934193349852946f;
        
        template <typename scalar_t, int ITEMS_PER_THREAD = 8>
        __global__ void selu_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            const int size) {
            
            // Grid-stride loop
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            const int stride = blockDim.x * gridDim.x;
            const int items_per_stride = stride * ITEMS_PER_THREAD;
            
            // Process multiple elements per thread
            for (int base = tid * ITEMS_PER_THREAD; base < size; base += items_per_stride) {
                scalar_t values[ITEMS_PER_THREAD];
                scalar_t results[ITEMS_PER_THREAD];
                
                // Load data - ensures coalesced memory access
                #pragma unroll
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    const int idx = base + i;
                    values[i] = (idx < size) ? input[idx] : 0;
                }
                
                // Process data
                #pragma unroll
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    const scalar_t x = values[i];
                    // Use ternary operator to minimize thread divergence
                    results[i] = SCALE * (x > 0 ? x : ALPHA * (__expf(x) - 1.0f));
                }
                
                // Store results - ensures coalesced memory access
                #pragma unroll
                for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                    const int idx = base + i;
                    if (idx < size) {
                        output[idx] = results[i];
                    }
                }
            }
        }
        
        // Kernel launcher
        torch::Tensor selu_cuda_forward(torch::Tensor input) {
            auto output = torch::empty_like(input);
            const int size = input.numel();
            
            const int threads = 256;
            const int max_blocks = 1024;
            // Calculate optimal number of blocks based on tensor size and items per thread
            const int blocks = min(max_blocks, (size + threads * 8 - 1) / (threads * 8));
            
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_kernel", ([&] {
                selu_kernel<scalar_t><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    size);
            }));
            
            return output;
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &selu_cuda_forward, "SELU forward (CUDA)");
        }
        """
        
        # Try to load the CUDA extension
        self.has_cuda_ext = False
        if torch.cuda.is_available():
            try:
                self.selu_cuda = torch.utils.cpp_extension.load_inline(
                    name="selu_optimized",
                    cpp_sources="",  # No separate C++ source needed
                    cuda_sources=cuda_source,
                    functions=["forward"],
                    with_cuda=True,
                    verbose=False
                )
                self.has_cuda_ext = True
            except Exception as e:
                print(f"Warning: CUDA extension compilation failed: {e}")
                self.has_cuda_ext = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies SELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with SELU applied, same shape as input.
        """
        # Use our custom CUDA kernel if available and input is on CUDA
        if self.has_cuda_ext and x.is_cuda:
            return self.selu_cuda.forward(x)
        else:
            # Fallback to PyTorch's implementation
            return torch.selu(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed