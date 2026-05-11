import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        bias (bool): Whether to use bias in the linear layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.in_features = in_features
        self.out_features = out_features
        
        # Pre-compute transposed weight for faster matrix multiplication
        self.register_buffer('weight_t', self.gemm.weight.t().contiguous())
        
        # Pre-compute bias minus subtract for efficiency
        if bias and self.gemm.bias is not None:
            self.register_buffer('bias_minus_subtract', self.gemm.bias - self.subtract)
        else:
            self.register_buffer('bias_minus_subtract', -self.subtract)
        
        # Pre-allocate buffers for intermediate results
        self.register_buffer('gemm_output', torch.zeros(batch_size, out_features))
        self.register_buffer('mean_output', torch.zeros(batch_size, 1))
        
        # Register parameter update hooks
        def update_weight_t(grad):
            if self.training:
                with torch.no_grad():
                    self.weight_t.copy_(self.gemm.weight.t().contiguous())
            return grad
        
        def update_bias_subtract(grad):
            if self.training:
                with torch.no_grad():
                    if hasattr(self.gemm, 'bias') and self.gemm.bias is not None:
                        self.bias_minus_subtract.copy_(self.gemm.bias - self.subtract)
                    else:
                        self.bias_minus_subtract.copy_(-self.subtract)
            return grad
        
        self.gemm.weight.register_hook(update_weight_t)
        if bias and self.gemm.bias is not None:
            self.gemm.bias.register_hook(update_bias_subtract)
        self.subtract.register_hook(update_bias_subtract)
        
        # CUDA kernel for fused operations
        if torch.cuda.is_available():
            self.cuda_kernel = self._load_kernel()
        else:
            self.cuda_kernel = None
    
    def _load_kernel(self):
        """Load the CUDA kernel for fused operations"""
        cuda_kernel = """
        extern "C" __global__ void fused_gemm_ops(
            const float* __restrict__ input,
            const float* __restrict__ weight_t,
            const float* __restrict__ bias_minus_subtract,
            float* __restrict__ output,
            const int batch_size,
            const int in_features,
            const int out_features)
        {
            // Block size parameters
            const int BLOCK_SIZE_M = 32;  // batch dimension
            const int BLOCK_SIZE_N = 32;  // output_features dimension
            const int BLOCK_SIZE_K = 32;  // inner product dimension
            
            // Thread coarsening factors - each thread computes a 4x4 output block
            const int TM = 4;
            const int TN = 4;
            
            // Shared memory for tiling with padding to avoid bank conflicts
            __shared__ float s_input[BLOCK_SIZE_M][BLOCK_SIZE_K + 1];
            __shared__ float s_weight[BLOCK_SIZE_K][BLOCK_SIZE_N + 1];
            
            // Block indices
            const int bx = blockIdx.x;
            const int by = blockIdx.y;
            
            // Thread indices
            const int tx = threadIdx.x;
            const int ty = threadIdx.y;
            
            // Thread ID within block
            const int tid = ty * blockDim.x + tx;
            
            // Shared memory for row sums (used for global average pooling)
            __shared__ float row_sums[BLOCK_SIZE_M];
            if (tx == 0) {
                row_sums[ty] = 0.0f;
            }
            
            // Register arrays for accumulating results
            float acc[TM][TN];
            
            // Initialize accumulators
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] = 0.0f;
                }
            }
            
            __syncthreads();
            
            // Loop over tiles
            for (int t = 0; t < (in_features + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++t) {
                // Collaborative loading of input tile using vectorized loads where possible
                for (int i = 0; i < BLOCK_SIZE_M; i += blockDim.y) {
                    if (i + ty < BLOCK_SIZE_M) {
                        const int row = by * BLOCK_SIZE_M + i + ty;
                        
                        // Use vectorized loads (float4) when possible
                        if (tx % 4 == 0 && t * BLOCK_SIZE_K + tx + 3 < in_features && row < batch_size) {
                            const int col = t * BLOCK_SIZE_K + tx;
                            float4 tmp = reinterpret_cast<const float4*>(&input[row * in_features + col])[0];
                            s_input[i + ty][tx] = tmp.x;
                            if (tx + 1 < BLOCK_SIZE_K) s_input[i + ty][tx + 1] = tmp.y;
                            if (tx + 2 < BLOCK_SIZE_K) s_input[i + ty][tx + 2] = tmp.z;
                            if (tx + 3 < BLOCK_SIZE_K) s_input[i + ty][tx + 3] = tmp.w;
                        }
                        else {
                            // Regular loading for edge cases
                            for (int j = 0; j < BLOCK_SIZE_K; j += blockDim.x) {
                                if (j + tx < BLOCK_SIZE_K) {
                                    const int col = t * BLOCK_SIZE_K + j + tx;
                                    if (row < batch_size && col < in_features) {
                                        s_input[i + ty][j + tx] = input[row * in_features + col];
                                    } else {
                                        s_input[i + ty][j + tx] = 0.0f;
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Collaborative loading of weight tile
                for (int i = 0; i < BLOCK_SIZE_K; i += blockDim.y) {
                    if (i + ty < BLOCK_SIZE_K) {
                        for (int j = 0; j < BLOCK_SIZE_N; j += blockDim.x) {
                            if (j + tx < BLOCK_SIZE_N) {
                                const int row = t * BLOCK_SIZE_K + i + ty;
                                const int col = bx * BLOCK_SIZE_N + j + tx;
                                if (row < in_features && col < out_features) {
                                    s_weight[i + ty][j + tx] = weight_t[col * in_features + row];
                                } else {
                                    s_weight[i + ty][j + tx] = 0.0f;
                                }
                            }
                        }
                    }
                }
                
                __syncthreads();
                
                // Compute partial dot products with register blocking
                #pragma unroll
                for (int k = 0; k < BLOCK_SIZE_K; ++k) {
                    // Load a row of input values
                    float a_vals[TM];
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        if (ty * TM + i < BLOCK_SIZE_M) {
                            a_vals[i] = s_input[ty * TM + i][k];
                        } else {
                            a_vals[i] = 0.0f;
                        }
                    }
                    
                    // Load a column of weight values
                    float b_vals[TN];
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        if (tx * TN + j < BLOCK_SIZE_N) {
                            b_vals[j] = s_weight[k][tx * TN + j];
                        } else {
                            b_vals[j] = 0.0f;
                        }
                    }
                    
                    // Compute outer product
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        #pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            acc[i][j] += a_vals[i] * b_vals[j];
                        }
                    }
                }
                
                __syncthreads();
            }
            
            // Add bias and perform subtract
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int row = by * BLOCK_SIZE_M + ty * TM + i;
                if (row < batch_size) {
                    float row_sum = 0.0f;
                    
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        const int col = bx * BLOCK_SIZE_N + tx * TN + j;
                        if (col < out_features) {
                            // Add bias and subtract
                            acc[i][j] += bias_minus_subtract[col];
                            row_sum += acc[i][j];
                        }
                    }
                    
                    // Contribute to row sum for global average pooling
                    atomicAdd(&row_sums[ty * TM + i], row_sum);
                }
            }
            
            __syncthreads();
            
            // Compute final operations (only one thread per row)
            if (tx == 0) {
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const int row = by * BLOCK_SIZE_M + ty * TM + i;
                    if (row < batch_size) {
                        // Global average pooling
                        float avg = row_sums[ty * TM + i] / out_features;
                        
                        // LogSumExp of a single value is just the value itself
                        float logsumexp_val = avg;
                        
                        // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                        const float sqrt_2_over_pi = 0.7978845608028654f;
                        float x3 = logsumexp_val * logsumexp_val * logsumexp_val;
                        float gelu_val = logsumexp_val * 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (logsumexp_val + 0.044715f * x3)));
                        
                        // ResidualAdd - add to original input
                        for (int j = 0; j < in_features; ++j) {
                            if (j == 0) {
                                output[row * in_features + j] = input[row * in_features + j] + gelu_val;
                            } else {
                                output[row * in_features + j] = input[row * in_features + j];
                            }
                        }
                    }
                }
            }
        }
        """
        
        try:
            from torch.utils.cpp_extension import load_inline
            
            return load_inline(
                name="fused_gemm_ops_cuda",
                cpp_sources="",
                cuda_sources=cuda_kernel,
                functions=["fused_gemm_ops"],
                with_cuda=True,
                verbose=False,
                extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++14"]
            )
        except Exception as e:
            print(f"Failed to load CUDA kernel: {e}")
            return None
    
    def forward(self, x):
        """
        Optimized forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor after all operations
        """
        # Store reference to original input (no clone needed)
        original_x = x
        
        # Ensure input is contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        
        batch_size_actual = x.size(0)
        
        # Try to use CUDA kernel for inference if available and on GPU
        if (not self.training and 
            self.cuda_kernel is not None and 
            x.is_cuda and 
            batch_size_actual == batch_size):
            
            # Allocate output tensor
            output = torch.empty_like(original_x)
            
            # Launch kernel
            grid_dim = (
                (self.out_features + 31) // 32,
                (batch_size_actual + 31) // 32
            )
            block_dim = (8, 8)  # 8x8=64 threads per block, each handling 4x4 elements
            
            self.cuda_kernel.fused_gemm_ops(
                x,
                self.weight_t,
                self.bias_minus_subtract,
                output,
                batch_size_actual,
                self.in_features,
                self.out_features,
                grid=grid_dim,
                block=block_dim
            )
            
            return output
        
        # Optimized PyTorch fallback path
        if batch_size_actual == batch_size and x.device == self.gemm_output.device:
            # Optimized GEMM operation using pre-transposed weights
            torch.addmm(self.bias_minus_subtract, x, self.weight_t, out=self.gemm_output)
            
            # GlobalAvgPool
            torch.mean(self.gemm_output, dim=1, keepdim=True, out=self.mean_output)
            
            # LogSumExp (for a single value per batch, this is just the value itself)
            # GELU
            x = torch.nn.functional.gelu(self.mean_output)
            
            # ResidualAdd
            return x + original_x
        else:
            # General fallback path
            x = self.gemm(x)
            x = x - self.subtract
            x = torch.mean(x, dim=1, keepdim=True)
            x = torch.logsumexp(x, dim=1, keepdim=True)
            x = torch.nn.functional.gelu(x)
            return x + original_x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]