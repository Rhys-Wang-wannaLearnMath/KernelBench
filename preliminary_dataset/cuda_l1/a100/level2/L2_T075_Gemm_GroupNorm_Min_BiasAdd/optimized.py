import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation of the model that performs GEMM, Group Normalization,
    Minimum operation, and Bias addition.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        num_groups (int): Number of groups for GroupNorm
        bias_shape (tuple): Shape of the bias tensor
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        # Initialize with the same components as the reference implementation
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store dimensions for reshaping operations
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        
        # Ensure all parameters are contiguous for optimal memory access
        self.gemm.weight.data = self.gemm.weight.data.contiguous()
        if self.gemm.bias is not None:
            self.gemm.bias.data = self.gemm.bias.data.contiguous()
        self.group_norm.weight.data = self.group_norm.weight.data.contiguous()
        self.group_norm.bias.data = self.group_norm.bias.data.contiguous()
        self.bias.data = self.bias.data.contiguous()
        
        # CUDA graph related attributes
        self.static_input = None
        self.graph = None
        self.static_output = None
        self.warmup_done = False
        self.last_input_shape = None
        self.use_cuda_graph = True
        
        # Compile the forward function if torch.compile is available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.optimized_forward = torch.compile(self._forward, fullgraph=True, backend="inductor")
        else:
            self.optimized_forward = self._forward
    
    def _forward(self, x):
        # Step 1: GEMM operation
        x = self.gemm(x)
        
        # Step 2: Group Normalization
        # Handle different input dimensions
        orig_shape = x.shape
        if x.dim() == 2:
            batch_size, features = x.shape
            # Use view instead of reshape to avoid memory copy when possible
            x = x.view(batch_size, features, 1, 1)
            x = self.group_norm(x)
            x = x.view(batch_size, features)
        else:
            x = self.group_norm(x)
        
        # Step 3: Min operation - use torch.amin for better performance
        # amin doesn't compute indices, making it potentially faster
        x = torch.amin(x, dim=1, keepdim=True)
        
        # Step 4: Bias addition
        x = x + self.bias
        
        return x
    
    def forward(self, x):
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Use CUDA graphs for repeated forward passes with same input shape
        if torch.cuda.is_available() and x.is_cuda and self.use_cuda_graph:
            current_shape = x.shape
            
            # If input shape changed or first run, reset graph
            if self.last_input_shape != current_shape:
                self.static_input = None
                self.graph = None
                self.static_output = None
                self.warmup_done = False
                self.last_input_shape = current_shape
            
            try:
                if not self.warmup_done:
                    # Do exactly 5 warmup iterations (optimal based on previous attempts)
                    for _ in range(5):
                        _ = self.optimized_forward(x)
                    
                    # Ensure CUDA operations are synchronized before capturing graph
                    torch.cuda.synchronize()
                    self.warmup_done = True
                    
                    # Initialize CUDA graph
                    self.static_input = torch.zeros_like(x, device=x.device)
                    self.graph = torch.cuda.CUDAGraph()
                    
                    # Capture the graph
                    with torch.cuda.graph(self.graph):
                        self.static_input.copy_(x)
                        self.static_output = self.optimized_forward(self.static_input)
                
                # Run the captured graph with new input data
                self.static_input.copy_(x)
                self.graph.replay()
                return self.static_output
                
            except Exception:
                # If CUDA graph fails for any reason, fall back to regular execution
                self.use_cuda_graph = False
        
        # Fallback for CPU or when CUDA is not available or CUDA graph failed
        return self.optimized_forward(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 512
out_features = 256
num_groups = 8
bias_shape = (1, out_features, 1, 1)

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_features, out_features, num_groups, bias_shape]