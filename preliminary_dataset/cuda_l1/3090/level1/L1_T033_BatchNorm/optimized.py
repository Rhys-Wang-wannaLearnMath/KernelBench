import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized model that performs Batch Normalization.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
        """
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)
        
        # Cache parameters for direct F.batch_norm call to avoid attribute lookup overhead
        self.weight = self.bn.weight
        self.bias = self.bn.bias
        self.running_mean = self.bn.running_mean
        self.running_var = self.bn.running_var
        self.eps = self.bn.eps
        self.momentum = self.bn.momentum
        self.num_features = num_features
        
        # For JIT compilation
        self._compiled_forward_train = None
        self._compiled_forward_eval = None
        
        # Flag to track if we've done a warmup pass
        self._warmup_done = False
        
        # Custom CUDA kernel for batch normalization
        if torch.cuda.is_available():
            self._setup_cuda_kernel()
    
    def _setup_cuda_kernel(self):
        """
        Set up custom CUDA kernel for batch normalization
        """
        self.cuda_kernel_code = """
        extern "C" __global__ void batch_norm_forward(
            const float* input, float* output,
            const float* weight, const float* bias,
            const float* running_mean, const float* running_var,
            int batch_size, int channels, int height, int width,
            float epsilon) {
            
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_size = batch_size * channels * height * width;
            
            if (idx >= total_size) return;
            
            int c = (idx / (height * width)) % channels;
            
            float mean = running_mean[c];
            float var = running_var[c];
            float gamma = weight[c];
            float beta = bias[c];
            
            float norm_factor = rsqrtf(var + epsilon);
            
            output[idx] = gamma * (input[idx] - mean) * norm_factor + beta;
        }
        """
        
        try:
            from torch.utils.cpp_extension import load_inline
            self.batch_norm_cuda = load_inline(
                name="batch_norm_cuda",
                cpp_sources="",
                cuda_sources=self.cuda_kernel_code,
                functions=["batch_norm_forward"],
                with_cuda=True,
                verbose=False
            )
            self.use_custom_kernel = True
        except Exception:
            # Fall back to PyTorch's implementation if custom kernel fails to load
            self.use_custom_kernel = False

    def _custom_batch_norm(self, x):
        """
        Apply batch normalization using our custom CUDA kernel
        """
        output = torch.empty_like(x)
        
        # Get tensor dimensions
        batch_size, channels, height, width = x.shape
        
        # Calculate grid and block dimensions for CUDA kernel
        threads_per_block = 1024
        blocks_per_grid = (batch_size * channels * height * width + threads_per_block - 1) // threads_per_block
        
        # Launch the kernel
        self.batch_norm_cuda.batch_norm_forward(
            blocks_per_grid, threads_per_block,
            (x, output, self.weight, self.bias, self.running_mean, self.running_var,
             batch_size, channels, height, width, self.eps)
        )
        
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Ensure all tensors are on the same device
        if x.is_cuda and self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)
            self.weight = self.weight.to(x.device)
            self.bias = self.bias.to(x.device)
        
        # Perform warmup if not done yet and we're on CUDA
        if not self._warmup_done and x.is_cuda:
            with torch.no_grad():
                # Warm up training mode
                self.bn.training = True
                _ = F.batch_norm(
                    x.clone(),
                    self.running_mean,
                    self.running_var,
                    self.weight,
                    self.bias,
                    True,
                    self.momentum,
                    self.eps
                )
                
                # Warm up evaluation mode
                self.bn.training = False
                _ = F.batch_norm(
                    x.clone(),
                    self.running_mean,
                    self.running_var,
                    self.weight,
                    self.bias,
                    False,
                    self.momentum,
                    self.eps
                )
                
                # Create JIT compiled versions for both modes
                try:
                    def _optimized_bn_train(input_tensor, running_mean, running_var, weight, bias, momentum, eps):
                        return F.batch_norm(
                            input_tensor,
                            running_mean,
                            running_var,
                            weight,
                            bias,
                            True,
                            momentum,
                            eps
                        )
                        
                    def _optimized_bn_eval(input_tensor, running_mean, running_var, weight, bias, momentum, eps):
                        return F.batch_norm(
                            input_tensor,
                            running_mean,
                            running_var,
                            weight,
                            bias,
                            False,
                            momentum,
                            eps
                        )
                    
                    self._compiled_forward_train = torch.jit.script(_optimized_bn_train)
                    self._compiled_forward_eval = torch.jit.script(_optimized_bn_eval)
                except Exception:
                    # If compilation fails, we'll fall back to standard implementation
                    pass
                
                # If we have a custom kernel, warm it up too
                if hasattr(self, 'use_custom_kernel') and self.use_custom_kernel:
                    try:
                        _ = self._custom_batch_norm(x.clone())
                    except Exception:
                        self.use_custom_kernel = False
                
            self._warmup_done = True
        
        # Try using custom CUDA kernel if available and we're in eval mode
        if x.is_cuda and not self.bn.training and hasattr(self, 'use_custom_kernel') and self.use_custom_kernel:
            try:
                return self._custom_batch_norm(x)
            except Exception:
                # Fall back to PyTorch implementation if custom kernel fails
                pass
        
        # Use compiled forward if available
        if x.is_cuda:
            if self.bn.training and self._compiled_forward_train is not None:
                return self._compiled_forward_train(
                    x, self.running_mean, self.running_var, self.weight, self.bias, 
                    self.momentum, self.eps
                )
            elif not self.bn.training and self._compiled_forward_eval is not None:
                return self._compiled_forward_eval(
                    x, self.running_mean, self.running_var, self.weight, self.bias, 
                    self.momentum, self.eps
                )
        
        # Direct call to F.batch_norm as fallback
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.bn.training,
            self.momentum,
            self.eps
        )

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]