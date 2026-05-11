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
        
        # Create the GRU layer with the same parameters as the reference implementation
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
        
        # Register h0 as a buffer to ensure it's moved to the correct device with the model
        self.register_buffer('h0', torch.randn((num_layers, batch_size, hidden_size)))
        
        # CUDA graph optimization variables
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.static_h_n = None
        self.graph_initialized = False
        
        # Expected input shape based on batch_first parameter
        self.expected_shape = (batch_size, seq_len, input_size) if batch_first else (seq_len, batch_size, input_size)
    
    def _initialize_cuda_graph(self, x):
        """Initialize and capture CUDA graph for optimized execution"""
        try:
            # Create static input tensor with the same properties as the input
            self.static_input = torch.empty_like(x).contiguous()
            h0_device = self.h0.to(device=x.device, non_blocking=True)
            
            # Simple warmup to ensure kernels are compiled
            output, h_n = self.gru(self.static_input, h0_device)
            
            # Create static output tensors with the same properties as the output
            self.static_output = torch.empty_like(output).contiguous()
            self.static_h_n = torch.empty_like(h_n).contiguous()
            
            # Synchronize to ensure warmup is complete
            torch.cuda.synchronize()
            
            # Capture the graph
            self.graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self.graph):
                output, h_n = self.gru(self.static_input, h0_device)
                self.static_output.copy_(output)
                self.static_h_n.copy_(h_n)
            
            self.graph_initialized = True
            return True
        except Exception:
            # If graph capture fails, reset all graph-related variables
            self.graph = None
            self.static_input = None
            self.static_output = None
            self.static_h_n = None
            self.graph_initialized = False
            return False
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, 
                 otherwise (batch_size, seq_len, input_size)
        :return: output: The output features from the last layer of the GRU, for each t
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Check if we can use CUDA graphs
        can_use_graph = (
            torch.cuda.is_available() and 
            x.is_cuda and 
            x.shape == self.expected_shape
        )
        
        if can_use_graph:
            # Initialize graph if not already done
            if not self.graph_initialized:
                if not self._initialize_cuda_graph(x):
                    # If initialization fails, fall back to standard execution
                    h0_device = self.h0.to(device=x.device, non_blocking=True)
                    output, _ = self.gru(x, h0_device)
                    return output
            
            try:
                # Copy input data to static tensor
                self.static_input.copy_(x)
                
                # Replay the graph
                self.graph.replay()
                
                # Return the output
                return self.static_output
            except Exception:
                # Fall back to standard execution if graph replay fails
                h0_device = self.h0.to(device=x.device, non_blocking=True)
                output, _ = self.gru(x, h0_device)
                return output
        
        # Standard execution path (fallback)
        h0_device = self.h0.to(device=x.device, non_blocking=True)
        output, _ = self.gru(x, h0_device)
        return output

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.randn(seq_len, batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]