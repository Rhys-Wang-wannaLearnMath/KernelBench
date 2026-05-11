import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class ModelNew(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
    Optimized implementation with custom CUDA kernel for better performance.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Optimized chunk sizes for PyTorch fallback implementation
        self.q_chunk_size = 256
        self.kv_chunk_size = 512
        
        # Try to load CUDA kernel if available
        self.use_cuda_kernel = False
        if torch.cuda.is_available():
            try:
                self.relu_attention_cuda = self._load_cuda_kernel()
                self.use_cuda_kernel = True
            except Exception as e:
                print(f"Failed to load CUDA kernel: {e}")
                self.use_cuda_kernel = False

    def _load_cuda_kernel(self):
        """Load custom CUDA kernel for optimized ReLU self-attention"""
        from torch.utils.cpp_extension import load_inline
        
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        
        template <typename scalar_t>
        __global__ void relu_self_attention_kernel(
            const scalar_t* __restrict__ q,
            const scalar_t* __restrict__ k,
            const scalar_t* __restrict__ v,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int num_heads,
            const int seq_len,
            const int head_dim,
            const float scale) {
            
            // Block indices
            const int batch_head_idx = blockIdx.x;
            const int batch_idx = batch_head_idx / num_heads;
            const int head_idx = batch_head_idx % num_heads;
            const int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
            
            // Return if out of bounds
            if (batch_idx >= batch_size || query_idx >= seq_len)
                return;
            
            // Base pointer offset for this batch and head
            const int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
            
            // Shared memory to cache key and value vectors
            extern __shared__ char shared_memory[];
            scalar_t* k_shared = (scalar_t*)shared_memory;
            scalar_t* v_shared = (scalar_t*)(shared_memory + blockDim.x * head_dim * sizeof(scalar_t));
            
            // Thread's dimension index in the head_dim
            const int dim_idx = threadIdx.x;
            
            // Initialize output accumulator
            scalar_t out_val = 0.0f;
            
            // Load query vector for this query position
            scalar_t q_val[64];  // Assuming max head_dim is 64, adjust if needed
            if (dim_idx < head_dim) {
                for (int d = 0; d < head_dim; d++) {
                    q_val[d] = q[batch_head_offset + query_idx * head_dim + d] * scale;
                }
            }
            
            // Process keys and values in blocks
            // We only need to process keys up to the query position (causal mask)
            const int BLOCK_SIZE = blockDim.x;
            for (int key_block_start = 0; key_block_start <= query_idx; key_block_start += BLOCK_SIZE) {
                const int key_block_end = min(key_block_start + BLOCK_SIZE, query_idx + 1);
                const int valid_keys = key_block_end - key_block_start;
                
                // Collaboratively load K and V data into shared memory
                for (int i = threadIdx.y * blockDim.x + threadIdx.x; 
                     i < valid_keys * head_dim; 
                     i += blockDim.x * blockDim.y) {
                    const int key_offset = i / head_dim;
                    const int dim_offset = i % head_dim;
                    const int key_idx = key_block_start + key_offset;
                    
                    if (key_idx <= query_idx) { // Causal mask check
                        k_shared[key_offset * head_dim + dim_offset] = 
                            k[batch_head_offset + key_idx * head_dim + dim_offset];
                        v_shared[key_offset * head_dim + dim_offset] = 
                            v[batch_head_offset + key_idx * head_dim + dim_offset];
                    }
                }
                __syncthreads();
                
                // Each thread processes one dimension of the head_dim
                if (dim_idx < head_dim) {
                    // Compute attention scores and apply to values
                    for (int key_offset = 0; key_offset < valid_keys; key_offset++) {
                        const int key_idx = key_block_start + key_offset;
                        
                        if (key_idx <= query_idx) { // Causal mask check
                            // Compute dot product for this query-key pair
                            scalar_t score = 0.0f;
                            for (int d = 0; d < head_dim; d++) {
                                score += q_val[d] * k_shared[key_offset * head_dim + d];
                            }
                            
                            // Apply ReLU activation
                            if (score > 0.0f) {
                                // Apply attention to this dimension of the value
                                out_val += score * v_shared[key_offset * head_dim + dim_idx];
                            }
                        }
                    }
                }
                __syncthreads();
            }
            
            // Write output for this thread's dimension
            if (dim_idx < head_dim && query_idx < seq_len) {
                output[batch_head_offset + query_idx * head_dim + dim_idx] = out_val;
            }
        }
        
        // Optimized kernel for small head dimensions
        template <typename scalar_t>
        __global__ void relu_self_attention_small_head_kernel(
            const scalar_t* __restrict__ q,
            const scalar_t* __restrict__ k,
            const scalar_t* __restrict__ v,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int num_heads,
            const int seq_len,
            const int head_dim,
            const float scale) {
            
            // Block indices
            const int batch_head_idx = blockIdx.x;
            const int batch_idx = batch_head_idx / num_heads;
            const int head_idx = batch_head_idx % num_heads;
            const int query_start = blockIdx.y * blockDim.y;
            
            // Shared memory to cache Q, K, V data
            extern __shared__ char shared_memory[];
            scalar_t* q_shared = (scalar_t*)shared_memory;
            scalar_t* k_shared = (scalar_t*)(shared_memory + blockDim.y * head_dim * sizeof(scalar_t));
            scalar_t* v_shared = (scalar_t*)(shared_memory + (blockDim.y + seq_len) * head_dim * sizeof(scalar_t));
            
            // Base pointer offset for this batch and head
            const int batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
            
            // Thread indices
            const int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
            const int total_threads = blockDim.x * blockDim.y;
            
            // Collaboratively load Q data for this block
            for (int i = thread_idx; i < blockDim.y * head_dim; i += total_threads) {
                const int q_idx = i / head_dim;
                const int d_idx = i % head_dim;
                const int seq_idx = query_start + q_idx;
                
                if (seq_idx < seq_len) {
                    q_shared[q_idx * head_dim + d_idx] = 
                        q[batch_head_offset + seq_idx * head_dim + d_idx] * scale;
                }
            }
            
            // Process in chunks to avoid loading the entire K, V matrices
            const int CHUNK_SIZE = 32;
            for (int key_chunk = 0; key_chunk <= query_start + blockDim.y - 1; key_chunk += CHUNK_SIZE) {
                const int chunk_end = min(key_chunk + CHUNK_SIZE, seq_len);
                const int chunk_size = chunk_end - key_chunk;
                
                // Collaboratively load K, V data for this chunk
                for (int i = thread_idx; i < chunk_size * head_dim; i += total_threads) {
                    const int k_idx = i / head_dim;
                    const int d_idx = i % head_dim;
                    const int seq_idx = key_chunk + k_idx;
                    
                    if (seq_idx < seq_len) {
                        k_shared[k_idx * head_dim + d_idx] = 
                            k[batch_head_offset + seq_idx * head_dim + d_idx];
                        v_shared[k_idx * head_dim + d_idx] = 
                            v[batch_head_offset + seq_idx * head_dim + d_idx];
                    }
                }
                
                __syncthreads();
                
                // Each thread processes one query position
                const int query_idx = query_start + threadIdx.y;
                if (query_idx < seq_len && threadIdx.x < head_dim) {
                    const int dim_idx = threadIdx.x;
                    
                    // Only process keys up to the query position (causal mask)
                    const int max_key_idx = min(chunk_end, query_idx + 1);
                    
                    for (int key_idx = key_chunk; key_idx < max_key_idx; key_idx++) {
                        // Compute dot product for this query-key pair
                        scalar_t score = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            score += q_shared[(query_idx - query_start) * head_dim + d] * 
                                    k_shared[(key_idx - key_chunk) * head_dim + d];
                        }
                        
                        // Apply ReLU activation
                        if (score > 0.0f) {
                            // Apply attention to this dimension of the value
                            output[batch_head_offset + query_idx * head_dim + dim_idx] += 
                                score * v_shared[(key_idx - key_chunk) * head_dim + dim_idx];
                        }
                    }
                }
                
                __syncthreads();
            }
        }
        
        torch::Tensor relu_self_attention_cuda(
            torch::Tensor q,
            torch::Tensor k,
            torch::Tensor v,
            float scale) {
            
            // Get dimensions
            const int batch_size = q.size(0);
            const int num_heads = q.size(1);
            const int seq_len = q.size(2);
            const int head_dim = q.size(3);
            
            // Create output tensor
            auto output = torch::zeros_like(q);
            
            // Choose kernel and configuration based on head dimension
            if (head_dim <= 32) {
                // For small head dimensions, use the optimized kernel
                const int threads_x = head_dim;
                const int threads_y = 16;
                const dim3 threads(threads_x, threads_y);
                const dim3 blocks(batch_size * num_heads, (seq_len + threads_y - 1) / threads_y);
                
                // Shared memory size: space for Q, K, V data
                const int shared_mem_size = (threads_y + 2 * seq_len) * head_dim * sizeof(float);
                
                AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "relu_self_attention_small_head_kernel", ([&] {
                    relu_self_attention_small_head_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
                        q.data_ptr<scalar_t>(),
                        k.data_ptr<scalar_t>(),
                        v.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_size,
                        num_heads,
                        seq_len,
                        head_dim,
                        scale
                    );
                }));
            } else {
                // For larger head dimensions, use the standard kernel
                const int threads_x = 32;
                const int threads_y = 8;
                const dim3 threads(threads_x, threads_y);
                const dim3 blocks(batch_size * num_heads, (seq_len + threads_y - 1) / threads_y);
                
                // Shared memory size: space for K and V data
                const int shared_mem_size = 2 * threads_x * head_dim * sizeof(float);
                
                AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "relu_self_attention_kernel", ([&] {
                    relu_self_attention_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
                        q.data_ptr<scalar_t>(),
                        k.data_ptr<scalar_t>(),
                        v.data_ptr<scalar_t>(),
                        output.data_ptr<scalar_t>(),
                        batch_size,
                        num_heads,
                        seq_len,
                        head_dim,
                        scale
                    );
                }));
            }
            
            return output;
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &relu_self_attention_cuda, "ReLU Self Attention forward (CUDA)");
        }
        """
        
        return load_inline(
            name="relu_attention_cuda",
            cpp_sources="",
            cuda_sources=cuda_source,
            functions=["forward"],
            verbose=True
        )

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Efficient QKV projection and reshaping
        qkv = self.c_attn(x)  # (B, T, 3*C)
        
        # Reshape qkv to separate q, k, v with minimal reshaping operations
        qkv = qkv.reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hs)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, nh, T, hs)
        
        # Ensure tensors are contiguous for efficient operations
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Use CUDA kernel if available and input is on CUDA
        if self.use_cuda_kernel and x.is_cuda:
            try:
                y = self.relu_attention_cuda.forward(q, k, v, self.scale)
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                return y
            except Exception as e:
                print(f"CUDA kernel execution failed: {e}, falling back to PyTorch implementation")
        
        # Fall back to optimized PyTorch implementation
        return self._forward_pytorch(q, k, v, B, T, C)
    
    def _forward_pytorch(self, q, k, v, B, T, C):
        """Optimized PyTorch implementation as fallback"""
        # Pre-allocate output tensor
        y = torch.zeros_like(q)
        
        # Process query sequence in chunks
        for i in range(0, T, self.q_chunk_size):
            i_end = min(i + self.q_chunk_size, T)
            q_chunk = q[:, :, i:i_end]  # (B, nh, chunk_size, hs)
            
            # Process key-value sequence in chunks up to current position
            for j in range(0, i_end, self.kv_chunk_size):
                j_end = min(j + self.kv_chunk_size, i_end)
                k_chunk = k[:, :, j:j_end]  # (B, nh, chunk_size, hs)
                v_chunk = v[:, :, j:j_end]  # (B, nh, chunk_size, hs)
                
                # Compute attention scores for this chunk pair
                att_chunk = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.scale
                
                # Apply causal mask - only needed for chunks where j+chunk_size > i
                if j + self.kv_chunk_size > i:
                    mask_chunk = self.bias[:, :, i:i_end, j:j_end]
                    att_chunk.masked_fill_(mask_chunk == 0, float('-inf'))
                
                # Apply ReLU activation
                att_chunk = F.relu(att_chunk)
                
                # Apply attention to values
                y[:, :, i:i_end] += torch.matmul(att_chunk, v_chunk)
        
        # Reshape output back to original format
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return y

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
max_seqlen = 1024
n_embd = 768  # Hidden dimension, typical for BERT-base size
n_head = 12   # Number of attention heads, typical for BERT-base size

def get_inputs():
    return [torch.randn(batch_size, max_seqlen, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]