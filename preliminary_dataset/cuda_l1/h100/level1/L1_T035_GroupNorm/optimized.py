import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Optimized CUDA kernel for GroupNorm
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for computing mean and variance using Welford's online algorithm
template <typename scalar_t>
__global__ void group_norm_stats_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var,
    int N, int C, int H, int W, int G) {
    
    // Each block handles one batch-group combination
    const int batch_idx = blockIdx.x / G;
    const int group_idx = blockIdx.x % G;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Calculate group parameters
    const int channels_per_group = C / G;
    const int HW = H * W;
    const int group_size = channels_per_group * HW;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = &sdata[block_size];
    
    // Initialize Welford's algorithm accumulators
    float local_mean = 0.0f;
    float local_m2 = 0.0f;
    int local_count = 0;
    
    // Process multiple elements per thread with stride access
    for (int i = tid; i < group_size; i += block_size) {
        // Calculate the actual index in the input tensor
        const int c_offset = i / HW;
        const int hw_offset = i % HW;
        const int c_idx = group_idx * channels_per_group + c_offset;
        const int input_idx = batch_idx * C * HW + c_idx * HW + hw_offset;
        
        const float val = static_cast<float>(input[input_idx]);
        
        // Welford's online algorithm for mean and variance
        local_count++;
        float delta = val - local_mean;
        local_mean += delta / local_count;
        float delta2 = val - local_mean;
        local_m2 += delta * delta2;
    }
    
    // Store in shared memory
    s_sum[tid] = local_mean * local_count; // sum
    s_sum_sq[tid] = local_m2;    // sum of squares adjusted
    __syncthreads();
    
    // Parallel reduction with sequential addressing to minimize bank conflicts
    for (int stride = block_size / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    // Final warp reduction using warp primitives for efficiency
    if (tid < 32) {
        // Unroll the last iterations
        if (block_size >= 64) {
            s_sum[tid] += s_sum[tid + 32];
            s_sum_sq[tid] += s_sum_sq[tid + 32];
        }
        // Use warp shuffle for the last iterations
        for (int offset = 16; offset > 0; offset >>= 1) {
            s_sum[tid] += __shfl_down_sync(0xffffffff, s_sum[tid], offset);
            s_sum_sq[tid] += __shfl_down_sync(0xffffffff, s_sum_sq[tid], offset);
        }
    }
    
    // Write final result
    if (tid == 0) {
        const float group_mean = s_sum[0] / group_size;
        const float group_var = s_sum_sq[0] / group_size;
        
        mean[batch_idx * G + group_idx] = static_cast<scalar_t>(group_mean);
        var[batch_idx * G + group_idx] = static_cast<scalar_t>(group_var);
    }
}

// CUDA kernel for applying normalization
template <typename scalar_t>
__global__ void group_norm_apply_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    int N, int C, int H, int W, int G, float eps) {
    
    // 2D grid: x-dimension is groups, y-dimension is batches
    const int group_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Calculate group parameters
    const int channels_per_group = C / G;
    const int HW = H * W;
    
    // Cache group statistics in shared memory
    __shared__ float s_mean, s_invstd;
    
    // Load statistics once per block
    if (tid == 0) {
        const int stats_idx = batch_idx * G + group_idx;
        s_mean = static_cast<float>(mean[stats_idx]);
        s_invstd = rsqrtf(static_cast<float>(var[stats_idx]) + eps);
    }
    __syncthreads();
    
    // Calculate the range of channels this block processes
    const int start_channel = group_idx * channels_per_group;
    const int end_channel = start_channel + channels_per_group;
    
    // Process elements in this channel group
    for (int c = start_channel; c < end_channel; c++) {
        // Load gamma and beta for this channel
        const float gamma_c = static_cast<float>(gamma[c]);
        const float beta_c = static_cast<float>(beta[c]);
        
        // Process elements in this channel with grid-stride loop
        for (int hw = tid; hw < HW; hw += block_size) {
            const int input_idx = batch_idx * C * HW + c * HW + hw;
            
            // Normalize and apply affine transformation
            const float val = static_cast<float>(input[input_idx]);
            const float normalized = (val - s_mean) * s_invstd;
            const float transformed = normalized * gamma_c + beta_c;
            
            output[input_idx] = static_cast<scalar_t>(transformed);
        }
    }
}

std::vector<torch::Tensor> group_norm_cuda_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    int num_groups,
    float eps) {
    
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const auto G = num_groups;
    
    auto output = torch::empty_like(input);
    auto mean = torch::empty({N, G}, input.options());
    auto var = torch::empty({N, G}, input.options());
    
    // Launch statistics kernel
    const int stats_threads = 256;
    const int stats_blocks = N * G;
    const size_t shared_mem_size = 2 * stats_threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "group_norm_stats_kernel", ([&] {
        group_norm_stats_kernel<scalar_t><<<stats_blocks, stats_threads, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>(),
            N, C, H, W, G);
    }));
    
    // Launch normalization kernel with 2D grid
    const int norm_threads = 256;
    dim3 norm_blocks(G, N);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "group_norm_apply_kernel", ([&] {
        group_norm_apply_kernel<scalar_t><<<norm_blocks, norm_threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            N, C, H, W, G, eps);
    }));
    
    return {output, mean, var};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &group_norm_cuda_forward, "GroupNorm forward (CUDA)");
}
"""

# Try to compile CUDA extension
try:
    group_norm_cuda = load_inline(
        name="group_norm_cuda",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["forward"],
        verbose=False
    )
    cuda_available = True
except Exception as e:
    print(f"CUDA compilation failed: {e}")
    cuda_available = False

class ModelNew(nn.Module):
    """
    Simple model that performs Group Normalization.
    """
    def __init__(self, num_features: int, num_groups: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(ModelNew, self).__init__()
        self.num_groups = num_groups
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5
        
        # Validate that channels can be divided into groups
        assert num_features % num_groups == 0, "num_features must be divisible by num_groups"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        # Use CUDA implementation if available and tensor is on GPU
        if cuda_available and x.is_cuda and x.dim() == 4:
            # Ensure tensor is contiguous for optimal memory access
            if not x.is_contiguous():
                x = x.contiguous()
            
            # Use optimized CUDA kernel
            result = group_norm_cuda.forward(x, self.weight, self.bias, self.num_groups, self.eps)
            return result[0]
        else:
            # Fallback to PyTorch's implementation
            return nn.functional.group_norm(
                x, self.num_groups, self.weight, self.bias, self.eps
            )

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features, num_groups]