import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(ModelNew, self).__init__()
        
        # Enable cuDNN benchmarking for automatic algorithm selection
        torch.backends.cudnn.benchmark = True
        
        # Set optimal workspace limit for cuDNN (4GB - balance between previous implementations)
        torch.backends.cudnn.workspace_limit = 4 * 1024 * 1024 * 1024
        
        # Create the GRU with optimized settings
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            bias, 
            batch_first, 
            dropout=0, 
            bidirectional=True
        )
        
        # Pre-allocate hidden state as buffer to avoid reallocation
        self.register_buffer('h0', torch.randn((num_layers * 2, batch_size, hidden_size)))
        
        # Store configuration for later use
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # For CUDA graphs optimization
        self.static_input = None
        self.static_h0 = None
        self.graph = None
        self.graph_output = None
        self.graph_hn = None
        self.use_cuda_graph = False
        self.warmup_done = False
        self.input_shape = None
        self.last_device = None
        
        # For TorchScript optimization
        self.scripted_gru = None
        self.use_script = False
        
        # Try to optimize with TorchScript for better performance
        self._optimize_with_torchscript()
    
    def _optimize_with_torchscript(self):
        """Optimize the GRU with TorchScript if possible"""
        try:
            # Create sample inputs for tracing
            sample_input = torch.zeros(
                (batch_size, seq_len, self.input_size) if self.batch_first 
                else (seq_len, batch_size, self.input_size),
                device='cpu'  # Start on CPU to avoid CUDA initialization issues
            )
            sample_h0 = torch.zeros(
                (self.num_layers * 2, batch_size, self.hidden_size),
                device='cpu'
            )
            
            # Define a function to trace that handles the GRU operation
            def gru_forward(x, h0):
                return self.gru(x, h0)
            
            # Create a scripted version of the GRU forward pass
            self.scripted_gru = torch.jit.trace(
                gru_forward,
                (sample_input, sample_h0),
                check_trace=False  # Disable trace checking for speed
            )
            
            # Optimize the script
            self.scripted_gru = torch.jit.optimize_for_inference(self.scripted_gru)
            
            self.use_script = True
        except Exception:
            # If optimization fails, continue with regular GRU
            self.use_script = False
            self.scripted_gru = None
    
    def _initialize_cuda_graph(self, x):
        """Initialize CUDA graph for repeated execution with same-sized inputs"""
        if not torch.cuda.is_available() or not x.is_cuda:
            return False
        
        try:
            # Only use CUDA graphs on supported GPUs (compute capability >= 7.0)
            major, _ = torch.cuda.get_device_capability(x.device)
            if major < 7:
                return False
            
            # Save input shape and device for future reference
            self.input_shape = x.shape
            self.last_device = x.device
            
            # Create static inputs for the graph
            self.static_input = torch.zeros_like(x, device=x.device)
            self.static_h0 = self.h0.clone().to(x.device)
            
            # Ensure static tensors are contiguous for optimal memory access
            if not self.static_input.is_contiguous():
                self.static_input = self.static_input.contiguous()
            if not self.static_h0.is_contiguous():
                self.static_h0 = self.static_h0.contiguous()
            
            # Pre-allocate output tensors
            if self.batch_first:
                output_shape = (batch_size, seq_len, self.hidden_size * 2)
            else:
                output_shape = (seq_len, batch_size, self.hidden_size * 2)
            
            self.graph_output = torch.zeros(output_shape, device=x.device)
            self.graph_hn = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size), device=x.device)
            
            # Optimal warmup to ensure all kernels are compiled (15 iterations)
            # This is a balance between No1 (20) and No2 (5)
            for _ in range(15):
                with torch.no_grad():
                    if self.use_script:
                        output, hn = self.scripted_gru(self.static_input, self.static_h0)
                    else:
                        output, hn = self.gru(self.static_input, self.static_h0)
            
            # Force synchronization before graph capture
            torch.cuda.synchronize()
            
            # Create CUDA graph
            self.graph = torch.cuda.CUDAGraph()
            
            # Capture the graph
            with torch.cuda.graph(self.graph):
                with torch.no_grad():
                    if self.use_script:
                        output, hn = self.scripted_gru(self.static_input, self.static_h0)
                    else:
                        output, hn = self.gru(self.static_input, self.static_h0)
                    self.graph_output.copy_(output)
                    self.graph_hn.copy_(hn)
            
            # Force synchronization to ensure graph is captured correctly
            torch.cuda.synchronize()
            
            return True
        except Exception:
            # If CUDA graph initialization fails, fall back to regular execution
            self.static_input = None
            self.static_h0 = None
            self.graph = None
            self.graph_output = None
            self.graph_hn = None
            self.input_shape = None
            self.last_device = None
            return False
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, 
                  otherwise (batch_size, seq_len, input_size)
        :return: output: The output features from the GRU
        """
        # Make sure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Move h0 to the same device as x if needed
        if self.h0.device != x.device:
            self.h0 = self.h0.to(x.device)
        
        # Ensure h0 is contiguous
        if not self.h0.is_contiguous():
            self.h0 = self.h0.contiguous()
        
        # Try using CUDA graphs for repeated execution with same-sized inputs
        if x.is_cuda:
            # Initialize CUDA graph if not done yet, input shape changed, or device changed
            if (not self.warmup_done or 
                (self.input_shape is not None and self.input_shape != x.shape) or
                (self.last_device is not None and self.last_device != x.device)):
                
                # If shape or device changed, we need to reinitialize the graph
                if self.warmup_done and (
                    (self.input_shape is not None and self.input_shape != x.shape) or 
                    (self.last_device is not None and self.last_device != x.device)
                ):
                    # Clean up old graph resources
                    self.static_input = None
                    self.static_h0 = None
                    self.graph = None
                    self.graph_output = None
                    self.graph_hn = None
                
                self.use_cuda_graph = self._initialize_cuda_graph(x)
                self.warmup_done = True
            
            # Use CUDA graph if available and input shape matches
            if self.use_cuda_graph and self.input_shape == x.shape and self.last_device == x.device:
                # Copy input data to static tensor
                self.static_input.copy_(x)
                
                # Run the captured graph
                self.graph.replay()
                
                # Return the output
                return self.graph_output
        
        # Fall back to standard execution if CUDA graph is not available or applicable
        with torch.no_grad():
            if self.use_script:
                output, _ = self.scripted_gru(x, self.h0)
            else:
                output, _ = self.gru(x, self.h0)
        
        return output

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.randn(seq_len, batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]