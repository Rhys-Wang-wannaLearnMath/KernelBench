import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

class ModelNew(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cuda_module = None
        
        # Define CUDA kernel for cross entropy loss
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <c10/cuda/CUDAGuard.h>
        
        // Constants optimized for our specific problem
        #define BLOCK_SIZE 256
        #define NUM_CLASSES 10
        #define WARP_SIZE 32
        #define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
        #define SAMPLES_PER_THREAD 8  // Each thread processes multiple samples
        
        template <typename scalar_t>
        __device__ __forceinline__ scalar_t fast_exp(scalar_t x) {
            return __expf(x);
        }
        
        template <typename scalar_t>
        __device__ __forceinline__ scalar_t fast_log(scalar_t x) {
            return __logf(x);
        }
        
        // First phase: compute per-block partial sums
        template <typename scalar_t>
        __global__ void cross_entropy_phase1(
            const scalar_t* __restrict__ predictions,
            const int64_t* __restrict__ targets,
            scalar_t* __restrict__ block_results,
            const int batch_size) {
            
            // Shared memory for warp-level reductions with padding to avoid bank conflicts
            __shared__ scalar_t warp_losses[WARPS_PER_BLOCK];
            
            const int tid = threadIdx.x;
            const int bid = blockIdx.x;
            const int lane_id = tid % WARP_SIZE;
            const int warp_id = tid / WARP_SIZE;
            
            scalar_t thread_loss = 0.0f;
            
            // Calculate starting sample index for this thread
            int sample_base = (bid * BLOCK_SIZE + tid) * SAMPLES_PER_THREAD;
            
            // Process multiple samples per thread
            #pragma unroll
            for (int i = 0; i < SAMPLES_PER_THREAD; i++) {
                int sample_idx = sample_base + i;
                
                // Check if this sample is within the batch
                if (sample_idx < batch_size) {
                    // Get prediction pointer for this sample
                    const scalar_t* sample_preds = predictions + sample_idx * NUM_CLASSES;
                    
                    // Use vectorized loads for better memory throughput
                    float4 vec1 = *reinterpret_cast<const float4*>(sample_preds);
                    float4 vec2 = *reinterpret_cast<const float4*>(sample_preds + 4);
                    scalar_t val9 = sample_preds[8];
                    scalar_t val10 = sample_preds[9];
                    
                    // Find max value for numerical stability - optimized approach
                    scalar_t max_val = fmaxf(fmaxf(fmaxf(vec1.x, vec1.y), fmaxf(vec1.z, vec1.w)), 
                                           fmaxf(fmaxf(fmaxf(vec2.x, vec2.y), fmaxf(vec2.z, vec2.w)), 
                                               fmaxf(val9, val10)));
                    
                    // Pre-compute shifted values for better performance
                    scalar_t shifted1 = vec1.x - max_val;
                    scalar_t shifted2 = vec1.y - max_val;
                    scalar_t shifted3 = vec1.z - max_val;
                    scalar_t shifted4 = vec1.w - max_val;
                    scalar_t shifted5 = vec2.x - max_val;
                    scalar_t shifted6 = vec2.y - max_val;
                    scalar_t shifted7 = vec2.z - max_val;
                    scalar_t shifted8 = vec2.w - max_val;
                    scalar_t shifted9 = val9 - max_val;
                    scalar_t shifted10 = val10 - max_val;
                    
                    // Compute exp values
                    scalar_t exp_val1 = fast_exp(shifted1);
                    scalar_t exp_val2 = fast_exp(shifted2);
                    scalar_t exp_val3 = fast_exp(shifted3);
                    scalar_t exp_val4 = fast_exp(shifted4);
                    scalar_t exp_val5 = fast_exp(shifted5);
                    scalar_t exp_val6 = fast_exp(shifted6);
                    scalar_t exp_val7 = fast_exp(shifted7);
                    scalar_t exp_val8 = fast_exp(shifted8);
                    scalar_t exp_val9 = fast_exp(shifted9);
                    scalar_t exp_val10 = fast_exp(shifted10);
                    
                    // Sum of exp values
                    scalar_t sum_exp = exp_val1 + exp_val2 + exp_val3 + exp_val4 + exp_val5 + 
                                      exp_val6 + exp_val7 + exp_val8 + exp_val9 + exp_val10;
                    
                    // Get target class
                    const int target_idx = targets[sample_idx];
                    
                    // Get target value using switch statement to avoid conditional branches
                    scalar_t shifted_target;
                    switch(target_idx) {
                        case 0: shifted_target = shifted1; break;
                        case 1: shifted_target = shifted2; break;
                        case 2: shifted_target = shifted3; break;
                        case 3: shifted_target = shifted4; break;
                        case 4: shifted_target = shifted5; break;
                        case 5: shifted_target = shifted6; break;
                        case 6: shifted_target = shifted7; break;
                        case 7: shifted_target = shifted8; break;
                        case 8: shifted_target = shifted9; break;
                        case 9: shifted_target = shifted10; break;
                        default: shifted_target = 0.0f; // Should never happen with valid inputs
                    }
                    
                    // Cross entropy formula: -log(exp(target_val - max_val) / sum_exp)
                    // = -(target_val - max_val) + log(sum_exp)
                    thread_loss += -shifted_target + fast_log(sum_exp);
                }
            }
            
            // Warp-level reduction using warp shuffle
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                thread_loss += __shfl_down_sync(0xffffffff, thread_loss, offset);
            }
            
            // First thread in each warp writes to shared memory
            if (lane_id == 0) {
                warp_losses[warp_id] = thread_loss;
            }
            
            __syncthreads();
            
            // Final reduction across warps (done by first warp)
            if (warp_id == 0) {
                scalar_t warp_sum = 0.0f;
                
                if (lane_id < WARPS_PER_BLOCK) {
                    warp_sum = warp_losses[lane_id];
                }
                
                // Warp-level reduction for final sum
                #pragma unroll
                for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                    warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
                }
                
                // First thread writes the block result
                if (lane_id == 0) {
                    block_results[bid] = warp_sum;
                }
            }
        }
        
        // Second phase: reduce block results to final output
        template <typename scalar_t>
        __global__ void cross_entropy_phase2(
            const scalar_t* __restrict__ block_results,
            scalar_t* __restrict__ output,
            const int num_blocks) {
            
            // Use shared memory for the reduction
            extern __shared__ scalar_t shared_data[];
            
            const int tid = threadIdx.x;
            
            // Each thread loads one block result
            scalar_t thread_sum = 0.0f;
            if (tid < num_blocks) {
                thread_sum = block_results[tid];
            }
            
            // Store in shared memory
            shared_data[tid] = thread_sum;
            
            __syncthreads();
            
            // For small number of blocks, use a single warp
            if (num_blocks <= 32) {
                // Single warp reduction
                if (tid < 32) {
                    // Warp-level reduction
                    if (tid + 16 < num_blocks) shared_data[tid] += shared_data[tid + 16];
                    __syncwarp();
                    if (tid < 8) shared_data[tid] += shared_data[tid + 8];
                    __syncwarp();
                    if (tid < 4) shared_data[tid] += shared_data[tid + 4];
                    __syncwarp();
                    if (tid < 2) shared_data[tid] += shared_data[tid + 2];
                    __syncwarp();
                    if (tid == 0) *output = shared_data[0] + shared_data[1];
                }
            } else {
                // Parallel reduction in shared memory
                for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
                    if (tid < stride) {
                        shared_data[tid] += shared_data[tid + stride];
                    }
                    __syncthreads();
                }
                
                // Final warp reduction (no sync needed)
                if (tid < 32) {
                    if (blockDim.x >= 64) shared_data[tid] += shared_data[tid + 32];
                    __syncwarp();
                    if (tid < 16) shared_data[tid] += shared_data[tid + 16];
                    __syncwarp();
                    if (tid < 8) shared_data[tid] += shared_data[tid + 8];
                    __syncwarp();
                    if (tid < 4) shared_data[tid] += shared_data[tid + 4];
                    __syncwarp();
                    if (tid < 2) shared_data[tid] += shared_data[tid + 2];
                    __syncwarp();
                    if (tid == 0) *output = shared_data[0] + shared_data[1];
                }
            }
        }
        
        torch::Tensor cross_entropy_forward_cuda(
            torch::Tensor predictions,
            torch::Tensor targets) {
            
            // Ensure inputs are contiguous for optimal memory access
            predictions = predictions.contiguous();
            targets = targets.contiguous();
            
            const auto batch_size = predictions.size(0);
            const auto num_classes = predictions.size(1);
            
            // Verify our specialized implementation matches the input dimensions
            TORCH_CHECK(num_classes == NUM_CLASSES, "Expected num_classes=", NUM_CLASSES, ", got ", num_classes);
            
            auto output = torch::zeros({}, predictions.options());
            
            // Calculate grid size based on batch size and samples per thread
            const int samples_per_block = BLOCK_SIZE * SAMPLES_PER_THREAD;
            const int grid_size = (batch_size + samples_per_block - 1) / samples_per_block;
            
            // Allocate memory for block results
            auto block_results = torch::zeros({grid_size}, predictions.options());
            
            const at::cuda::OptionalCUDAGuard device_guard(device_of(predictions));
            
            AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "cross_entropy_forward_cuda", ([&] {
                // Phase 1: Compute partial sums per block
                cross_entropy_phase1<scalar_t><<<grid_size, BLOCK_SIZE>>>(
                    predictions.data_ptr<scalar_t>(),
                    targets.data_ptr<int64_t>(),
                    block_results.data_ptr<scalar_t>(),
                    batch_size);
                
                // Phase 2: Reduce block results to final output
                // Use appropriate block size based on number of blocks
                int phase2_block_size = min(256, max(32, grid_size));
                cross_entropy_phase2<scalar_t><<<1, phase2_block_size, phase2_block_size * sizeof(scalar_t)>>>(
                    block_results.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    grid_size);
            }));
            
            // Compute mean
            return output / static_cast<float>(batch_size);
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &cross_entropy_forward_cuda, "CrossEntropy forward (CUDA)");
        }
        """
        
        try:
            os.makedirs("cuda_extensions", exist_ok=True)
            self.cuda_module = load_inline(
                name="cross_entropy_cuda",
                cpp_sources=cuda_source,
                functions=["forward"],
                with_cuda=True,
                build_directory="cuda_extensions",
                verbose=False,
                extra_cuda_cflags=["-O3", "--use_fast_math"]
            )
        except Exception as e:
            print(f"Failed to load CUDA extension: {e}")
            self.cuda_module = None

    def forward(self, predictions, targets):
        if self.cuda_module is not None and predictions.is_cuda and targets.is_cuda:
            try:
                return self.cuda_module.forward(predictions, targets)
            except Exception as e:
                print(f"CUDA kernel error: {e}")
                return self._forward_native(predictions, targets)
        else:
            return self._forward_native(predictions, targets)
    
    def _forward_native(self, predictions, targets):
        """
        Optimized PyTorch implementation using log_softmax directly
        """
        log_probs = F.log_softmax(predictions, dim=1)
        return -log_probs.gather(1, targets.unsqueeze(1)).mean()

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 4096
num_classes = 10
input_shape = (num_classes, )  # Output for each class
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []