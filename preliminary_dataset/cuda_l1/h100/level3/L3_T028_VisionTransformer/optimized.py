import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os
import pathlib

# --- CUDA Kernel JIT Compilation ---
# This section defines, builds, and loads ALL the best custom CUDA kernels identified:
# 1. Triple-Fusion LayerNorm
# 2. Two-Stage Vectorized Shared-Memory Patch Assembly
# 3. Fused Bias+ReLU for the FFN
def load_fused_kernels_extension():
    """
    Builds the CUDA extension JIT with the full suite of optimized kernels.
    """
    build_dir = pathlib.Path('./vit_fully_fused_build')
    build_dir.mkdir(exist_ok=True)

    # C++ source for PyTorch binding
    cpp_source = """
#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
torch::Tensor fused_triple_add_layernorm_forward_cuda(
    const torch::Tensor& matmul_result, const torch::Tensor& residual,
    const torch::Tensor& linear_bias, const torch::Tensor& gamma, const torch::Tensor& beta,
    double epsilon);

torch::Tensor fused_patch_assembly_vectorized_forward_cuda(
    const torch::Tensor& conv_out_flat, const torch::Tensor& cls_token, const torch::Tensor& pos_embedding,
    int H_out, int W_out);
    
torch::Tensor fused_bias_relu_forward_cuda(
    const torch::Tensor& input, const torch::Tensor& bias);


// C++ interface with input checks
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fused_triple_add_layernorm_forward(
    const torch::Tensor& matmul_result, const torch::Tensor& residual,
    const torch::Tensor& linear_bias, const torch::Tensor& gamma, const torch::Tensor& beta,
    double epsilon) {
    CHECK_INPUT(matmul_result); CHECK_INPUT(residual); CHECK_INPUT(linear_bias);
    CHECK_INPUT(gamma); CHECK_INPUT(beta);
    return fused_triple_add_layernorm_forward_cuda(matmul_result, residual, linear_bias, gamma, beta, epsilon);
}

torch::Tensor fused_patch_assembly_forward(
    const torch::Tensor& conv_out_flat, const torch::Tensor& cls_token, const torch::Tensor& pos_embedding,
    int H_out, int W_out) {
    CHECK_INPUT(conv_out_flat); CHECK_INPUT(cls_token); CHECK_INPUT(pos_embedding);
    return fused_patch_assembly_vectorized_forward_cuda(conv_out_flat, cls_token, pos_embedding, H_out, W_out);
}

torch::Tensor fused_bias_relu_forward(
    const torch::Tensor& input, const torch::Tensor& bias) {
    CHECK_INPUT(input); CHECK_INPUT(bias);
    return fused_bias_relu_forward_cuda(input, bias);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("triple_layernorm", &fused_triple_add_layernorm_forward, "Fused (Matmul Result + Residual + Bias + LayerNorm) forward (CUDA)");
    m.def("patch_assembly", &fused_patch_assembly_forward, "Vectorized Fused Patch Assembly with Shared Memory (CUDA)");
    m.def("bias_relu", &fused_bias_relu_forward, "Fused (Bias + ReLU) forward (CUDA)");
}
"""

    # CUDA kernel source
    cu_source = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

constexpr int WARP_SIZE = 32;

// --- KERNEL 1: Fused Triple Add + LayerNorm ---
template <typename T, int D>
__global__ void fused_triple_add_layernorm_kernel(
    const T* __restrict__ matmul_ptr, const T* __restrict__ residual_ptr, const T* __restrict__ bias_ptr,
    const T* __restrict__ gamma_ptr, const T* __restrict__ beta_ptr, T* __restrict__ out_ptr,
    const int N, const float epsilon) {
    const int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_idx >= N) return;

    const int lane_id = threadIdx.x;
    using VecT = float4;
    constexpr int VEC_SIZE = sizeof(VecT) / sizeof(T);
    constexpr int VEC_D = D / VEC_SIZE;
    constexpr int THREAD_VEC_COUNT = VEC_D / WARP_SIZE;

    const VecT* matmul_vec_ptr = reinterpret_cast<const VecT*>(matmul_ptr) + row_idx * VEC_D;
    const VecT* residual_vec_ptr = reinterpret_cast<const VecT*>(residual_ptr) + row_idx * VEC_D;
    const VecT* bias_vec_ptr = reinterpret_cast<const VecT*>(bias_ptr);
    const VecT* gamma_vec_ptr = reinterpret_cast<const VecT*>(gamma_ptr);
    const VecT* beta_vec_ptr = reinterpret_cast<const VecT*>(beta_ptr);
    VecT* out_vec_ptr = reinterpret_cast<VecT*>(out_ptr) + row_idx * VEC_D;

    VecT temp_storage[THREAD_VEC_COUNT];
    float thread_sum = 0.0f; float thread_sum_sq = 0.0f;

    #pragma unroll
    for (int i = 0; i < THREAD_VEC_COUNT; ++i) {
        const int vec_idx = lane_id + i * WARP_SIZE;
        const VecT matmul_val = matmul_vec_ptr[vec_idx];
        const VecT residual_val = residual_vec_ptr[vec_idx];
        const VecT bias_val = bias_vec_ptr[vec_idx];
        VecT sum_val;
        sum_val.x = static_cast<float>(matmul_val.x) + static_cast<float>(residual_val.x) + static_cast<float>(bias_val.x);
        sum_val.y = static_cast<float>(matmul_val.y) + static_cast<float>(residual_val.y) + static_cast<float>(bias_val.y);
        sum_val.z = static_cast<float>(matmul_val.z) + static_cast<float>(residual_val.z) + static_cast<float>(bias_val.z);
        sum_val.w = static_cast<float>(matmul_val.w) + static_cast<float>(residual_val.w) + static_cast<float>(bias_val.w);
        temp_storage[i] = sum_val;
        thread_sum += (sum_val.x + sum_val.y + sum_val.z + sum_val.w);
        thread_sum_sq += (sum_val.x * sum_val.x + sum_val.y * sum_val.y + sum_val.z * sum_val.z + sum_val.w * sum_val.w);
    }
    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
    }

    float mean, inv_stddev;
    if (lane_id == 0) {
        mean = thread_sum / D;
        float var = thread_sum_sq / D - mean * mean;
        inv_stddev = rsqrtf(var + epsilon);
    }
    mean = __shfl_sync(0xffffffff, mean, 0);
    inv_stddev = __shfl_sync(0xffffffff, inv_stddev, 0);
    
    #pragma unroll
    for (int i = 0; i < THREAD_VEC_COUNT; ++i) {
        const int vec_idx = lane_id + i * WARP_SIZE;
        const VecT sum_val = temp_storage[i];
        const VecT gamma_val = gamma_vec_ptr[vec_idx];
        const VecT beta_val = beta_vec_ptr[vec_idx];
        VecT out_val;
        out_val.x = static_cast<T>(((sum_val.x - mean) * inv_stddev) * static_cast<float>(gamma_val.x) + static_cast<float>(beta_val.x));
        out_val.y = static_cast<T>(((sum_val.y - mean) * inv_stddev) * static_cast<float>(gamma_val.y) + static_cast<float>(beta_val.y));
        out_val.z = static_cast<T>(((sum_val.z - mean) * inv_stddev) * static_cast<float>(gamma_val.z) + static_cast<float>(beta_val.z));
        out_val.w = static_cast<T>(((sum_val.w - mean) * inv_stddev) * static_cast<float>(gamma_val.w) + static_cast<float>(beta_val.w));
        out_vec_ptr[vec_idx] = out_val;
    }
}

// --- KERNEL 2: Fused Bias + ReLU for FFN ---
template <typename T>
__global__ void fused_bias_relu_kernel(
    T* __restrict__ out_ptr,
    const T* __restrict__ in_ptr,
    const T* __restrict__ bias_ptr,
    const int N, const int D) {

    using VecT = float4;
    constexpr int VEC_SIZE = sizeof(VecT) / sizeof(T);
    const int D_VEC = D / VEC_SIZE;
    const int total_vecs = N * D_VEC;

    const VecT* in_vec_ptr = reinterpret_cast<const VecT*>(in_ptr);
    const VecT* bias_vec_ptr = reinterpret_cast<const VecT*>(bias_ptr);
    VecT* out_vec_ptr = reinterpret_cast<VecT*>(out_ptr);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vecs; i += gridDim.x * blockDim.x) {
        const int d_vec = i % D_VEC;
        
        const VecT in_val = in_vec_ptr[i];
        const VecT bias_val = bias_vec_ptr[d_vec];
        
        VecT out_val;
        out_val.x = max(static_cast<T>(0.0), in_val.x + bias_val.x);
        out_val.y = max(static_cast<T>(0.0), in_val.y + bias_val.y);
        out_val.z = max(static_cast<T>(0.0), in_val.z + bias_val.z);
        out_val.w = max(static_cast<T>(0.0), in_val.w + bias_val.w);

        out_vec_ptr[i] = out_val;
    }
}

// --- KERNEL 3: Vectorized Two-Stage Fused Patch Assembly ---
constexpr int TILE_DIM = 16;
constexpr int THREADS_PER_BLOCK_PATCH = 256;

// Kernel 3a: Handles the CLS token, vectorized
template <typename T, typename VecT>
__global__ void add_cls_pos_vectorized_kernel(
    VecT* __restrict__ out_vec_ptr,
    const VecT* __restrict__ cls_token_vec_ptr,
    const VecT* __restrict__ pos_embedding_vec_ptr,
    const int B, const int S, const int D_VEC) {

    const int total_elements = B * D_VEC;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += gridDim.x * blockDim.x) {
        const int b = i / D_VEC;
        const int d_vec = i % D_VEC;
        
        const int out_idx = b * S * D_VEC + d_vec; // s=0
        const int pos_idx = d_vec; // pos_embedding for [CLS] is at the start
        
        out_vec_ptr[out_idx] = cls_token_vec_ptr[d_vec] + pos_embedding_vec_ptr[pos_idx];
    }
}

// Kernel 3b: Transposes patches with shared memory and adds positional embeddings, vectorized
template <typename T, typename VecT>
__global__ void transpose_patch_pos_vectorized_kernel(
    VecT* __restrict__ out_vec_ptr,
    const VecT* __restrict__ conv_out_flat_vec_ptr,
    const VecT* __restrict__ pos_embedding_vec_ptr,
    const int B, const int S, const int D_VEC, const int P) {
    
    __shared__ VecT tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    const int b = blockIdx.z;

    const int tile_p_base = blockIdx.x * TILE_DIM;
    const int tile_d_vec_base = blockIdx.y * TILE_DIM;

    const int p_in_tile = threadIdx.x;
    const int d_vec_in_tile = threadIdx.y;

    // Coalesced vector read from conv_out_flat (B, D_VEC, P) into shared memory
    const int src_p = tile_p_base + p_in_tile;
    const int src_d_vec = tile_d_vec_base + d_vec_in_tile;
    if (src_d_vec < D_VEC && src_p < P) {
        const int read_idx = b * D_VEC * P + src_d_vec * P + src_p;
        tile[d_vec_in_tile][p_in_tile] = conv_out_flat_vec_ptr[read_idx];
    }

    __syncthreads();

    // Coalesced vector write from shared memory to output (B, P, D_VEC)
    const int dst_p = tile_p_base + d_vec_in_tile;
    const int dst_d_vec = tile_d_vec_base + p_in_tile;

    if (dst_p < P && dst_d_vec < D_VEC) {
        const int s = dst_p + 1;
        const int out_idx = b * S * D_VEC + s * D_VEC + dst_d_vec;
        const int pos_idx = s * D_VEC + dst_d_vec;
        
        out_vec_ptr[out_idx] = tile[p_in_tile][d_vec_in_tile] + pos_embedding_vec_ptr[pos_idx];
    }
}

// --- CUDA Forward Pass Implementations ---

torch::Tensor fused_triple_add_layernorm_forward_cuda(
    const torch::Tensor& matmul_result, const torch::Tensor& residual, const torch::Tensor& linear_bias,
    const torch::Tensor& gamma, const torch::Tensor& beta, double epsilon) {
    const int N = matmul_result.numel() / matmul_result.size(-1);
    const int D = matmul_result.size(-1);
    TORCH_CHECK(D == 512, "Triple-fusion kernel is specialized for D=512.");
    auto out = torch::empty_like(residual);
    const int warps_per_block = 8;
    dim3 threads(WARP_SIZE, warps_per_block);
    dim3 blocks((N + warps_per_block - 1) / warps_per_block);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(matmul_result.scalar_type(), "fused_triple_add_layernorm", ([&] {
        fused_triple_add_layernorm_kernel<scalar_t, 512><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            matmul_result.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(), linear_bias.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), N, static_cast<float>(epsilon)
        );
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_bias_relu_forward_cuda(
    const torch::Tensor& input, const torch::Tensor& bias) {
    const int N = input.numel() / input.size(-1);
    const int D = input.size(-1);
    TORCH_CHECK(D % 4 == 0, "FusedBiasReLU requires dimension divisible by 4.");
    auto out = torch::empty_like(input);
    const int total_vecs = N * (D / 4);
    const int threads_per_block = 256;
    const int blocks = (total_vecs + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_bias_relu", ([&] {
        fused_bias_relu_kernel<scalar_t><<<blocks, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), N, D
        );
    }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_patch_assembly_vectorized_forward_cuda(
    const torch::Tensor& conv_out_flat, const torch::Tensor& cls_token, const torch::Tensor& pos_embedding,
    const int H_out, const int W_out) {
    
    const int B = conv_out_flat.size(0);
    const int D = conv_out_flat.size(1);
    const int P = H_out * W_out;
    const int S = P + 1;

    TORCH_CHECK(D % 4 == 0, "Dimension must be divisible by 4 for vectorization.");
    TORCH_CHECK(conv_out_flat.dim() == 3, "conv_out_flat must be a 3D tensor (B, D, P)");
    const int D_VEC = D / 4;

    auto out = torch::empty({B, S, D}, conv_out_flat.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(conv_out_flat.scalar_type(), "fused_patch_assembly_vectorized", ([&] {
        using T = scalar_t;
        using VecT = float4;

        const int cls_blocks = (B * D_VEC + THREADS_PER_BLOCK_PATCH - 1) / THREADS_PER_BLOCK_PATCH;
        add_cls_pos_vectorized_kernel<T, VecT><<<cls_blocks, THREADS_PER_BLOCK_PATCH, 0, stream>>>(
            reinterpret_cast<VecT*>(out.data_ptr<T>()),
            reinterpret_cast<const VecT*>(cls_token.data_ptr<T>()),
            reinterpret_cast<const VecT*>(pos_embedding.data_ptr<T>()), B, S, D_VEC);

        dim3 threads(TILE_DIM, TILE_DIM); // 16x16 = 256
        dim3 blocks((P + TILE_DIM - 1) / TILE_DIM, (D_VEC + TILE_DIM - 1) / TILE_DIM, B);
        transpose_patch_pos_vectorized_kernel<T, VecT><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<VecT*>(out.data_ptr<T>()),
            reinterpret_cast<const VecT*>(conv_out_flat.data_ptr<T>()),
            reinterpret_cast<const VecT*>(pos_embedding.data_ptr<T>()), B, S, D_VEC, P);
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
'''
    cpp_path = build_dir / "fused_ops.cpp"
    cu_path = build_dir / "fused_kernels.cu"
    with open(cpp_path, "w") as f: f.write(cpp_source)
    with open(cu_path, "w") as f: f.write(cu_source)

    try:
        verbose_build = os.environ.get('TORCH_EXTENSIONS_VERBOSE', '0') == '1'
        fused_module = load(
            name="vit_fully_fused_kernels",
            sources=[str(cpp_path), str(cu_path)],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            build_directory=str(build_dir),
            verbose=verbose_build
        )
        return fused_module
    except Exception as e:
        print(f"Failed to load CUDA extension: {e}")
        return None

fused_kernels = load_fused_kernels_extension()

class FusedTripleOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matmul_result, residual, linear_bias, norm_layer):
        if (fused_kernels and matmul_result.is_cuda and matmul_result.size(-1) == 512
                and matmul_result.is_contiguous() and residual.is_contiguous()):
            return fused_kernels.triple_layernorm(matmul_result, residual, linear_bias, norm_layer.weight, norm_layer.bias, norm_layer.eps)
        return F.layer_norm(residual + matmul_result + linear_bias, (residual.size(-1),), norm_layer.weight, norm_layer.bias, norm_layer.eps)

class FusedPatchAssembly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, conv_out_flat, cls_token, pos_embedding, H_out, W_out):
         if (fused_kernels and conv_out_flat.is_cuda and conv_out_flat.size(1) % 4 == 0):
             return fused_kernels.patch_assembly(conv_out_flat, cls_token, pos_embedding, H_out, W_out)
         x_patches = conv_out_flat.transpose(1, 2)
         cls_tokens = cls_token.expand(conv_out_flat.shape[0], -1, -1)
         x = torch.cat((cls_tokens, x_patches), dim=1)
         return x + pos_embedding

class FusedBiasReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, bias):
        if (fused_kernels and input_tensor.is_cuda and input_tensor.size(-1) % 4 == 0):
            return fused_kernels.bias_relu(input_tensor, bias)
        return F.relu(input_tensor + bias)

class CustomTransformerLayer(nn.Module):
    def __init__(self, ref_layer):
        super().__init__()
        dim = ref_layer.linear1.in_features
        self.in_proj_weight = nn.Parameter(ref_layer.self_attn.in_proj_weight.detach().clone())
        self.in_proj_bias = nn.Parameter(ref_layer.self_attn.in_proj_bias.detach().clone())
        
        out_proj_weight = ref_layer.self_attn.out_proj.weight.detach().clone()
        self.out_proj_weight_t = nn.Parameter(out_proj_weight.T.contiguous())
        self.out_proj_bias = nn.Parameter(ref_layer.self_attn.out_proj.bias.detach().clone())
        
        linear1_weight = ref_layer.linear1.weight.detach().clone()
        self.linear1_weight_t = nn.Parameter(linear1_weight.T.contiguous())
        self.linear1_bias = nn.Parameter(ref_layer.linear1.bias.detach().clone())

        linear2_weight = ref_layer.linear2.weight.detach().clone()
        self.linear2_weight_t = nn.Parameter(linear2_weight.T.contiguous())
        self.linear2_bias = nn.Parameter(ref_layer.linear2.bias.detach().clone())

        self.norm1 = nn.LayerNorm(dim, eps=ref_layer.norm1.eps)
        self.norm2 = nn.LayerNorm(dim, eps=ref_layer.norm2.eps)
        self.norm1.load_state_dict(ref_layer.norm1.state_dict())
        self.norm2.load_state_dict(ref_layer.norm2.state_dict())

# This class must be an EXACT copy of the reference to ensure weights are identical.
class Model(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(Model, self).__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True),
            num_layers=depth
        )
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, num_classes)
        )
    def forward(self, img):
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        super(ModelNew, self).__init__()
        self.patch_size = patch_size
        self.heads = heads
        self.channels = channels
        self.dim = dim

        ref_model = Model(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout)
        
        self.pos_embedding = nn.Parameter(ref_model.pos_embedding.detach().clone())
        self.cls_token = nn.Parameter(ref_model.cls_token.detach().clone())
        
        self.patch_to_embedding_weight = nn.Parameter(ref_model.patch_to_embedding.weight.detach().clone())
        self.patch_to_embedding_bias = nn.Parameter(ref_model.patch_to_embedding.bias.detach().clone())

        self.layers = nn.ModuleList([CustomTransformerLayer(ref_model.transformer.layers[i]) for i in range(depth)])
        
        self.mlp_linear1 = nn.Linear(dim, mlp_dim)
        self.mlp_linear2 = nn.Linear(mlp_dim, num_classes)
        self.mlp_linear1.load_state_dict(ref_model.mlp_head[0].state_dict())
        self.mlp_linear2.load_state_dict(ref_model.mlp_head[3].state_dict())
        
        self.graph = None
        self.static_input = None
        self.static_output = None

    def _forward_impl(self, img):
        B, C, H, W = img.shape
        H_out, W_out = H // self.patch_size, W // self.patch_size
        S_plus_1 = H_out * W_out + 1
        H_heads, D_h = self.heads, self.dim // self.heads

        conv_weight = self.patch_to_embedding_weight.view(self.dim, self.channels, self.patch_size, self.patch_size)
        conv_out = F.conv2d(img, conv_weight, self.patch_to_embedding_bias, stride=self.patch_size)
        
        x = FusedPatchAssembly.apply(conv_out.flatten(2), self.cls_token.squeeze(0), self.pos_embedding.squeeze(0), H_out, W_out)

        for layer in self.layers:
            residual_mha = x
            qkv = F.linear(x, layer.in_proj_weight, layer.in_proj_bias)
            
            qkv = qkv.view(B, S_plus_1, 3, H_heads, D_h).permute(2, 1, 3, 0, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn_output = F.scaled_dot_product_attention(q, k, v)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(B, S_plus_1, self.dim)
            
            attn_matmul_out = torch.matmul(attn_output, layer.out_proj_weight_t)
            x = FusedTripleOp.apply(attn_matmul_out, residual_mha, layer.out_proj_bias, layer.norm1)

            residual_ffn = x
            
            ffn_matmul1 = torch.matmul(x, layer.linear1_weight_t)
            ffn_inner = FusedBiasReLU.apply(ffn_matmul1, layer.linear1_bias)
            
            ffn_matmul_out = torch.matmul(ffn_inner, layer.linear2_weight_t)
            x = FusedTripleOp.apply(ffn_matmul_out, residual_ffn, layer.linear2_bias, layer.norm2)

        x = x[:, 0]
        
        x = self.mlp_linear1(x)
        x = F.gelu(x)
        x = self.mlp_linear2(x)
        return x

    def forward(self, img):
        if self.graph is None:
            self._forward_impl(img.clone()) 

            self.static_input = img.clone()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_input)

        self.static_input.copy_(img)
        self.graph.replay()
        return self.static_output.clone()

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(2, channels, image_size, image_size)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation  
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]