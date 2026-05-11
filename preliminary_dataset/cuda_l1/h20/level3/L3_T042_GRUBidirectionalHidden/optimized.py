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
        
        # Create the GRU module
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        
        # Pre-allocate the initial hidden state tensor
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        
        # CUDA graph optimization variables
        self.cuda_graph = None
        self.static_input = None
        self.static_h0 = None
        self.static_h_n = None
        self.graph_initialized = False
        self.input_shape = None
        self.last_device = None
    
    def _initialize_cuda_graph(self, x):
        """Initialize CUDA graph for the given input tensor"""
        # Store input shape and device for future reference
        self.input_shape = x.shape
        self.last_device = x.device
        
        # Move GRU to the same device as input
        self.gru = self.gru.to(x.device)
        
        # Create static tensors for graph capture
        self.static_input = torch.zeros_like(x, device=x.device)
        self.static_h0 = self.h0.to(x.device).clone()
        
        # Warm up with exactly 3 iterations (optimal from No2)
        for _ in range(3):
            with torch.no_grad():
                _, _ = self.gru(self.static_input, self.static_h0)
        
        # Create and capture the CUDA graph
        try:
            self.cuda_graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self.cuda_graph):
                _, self.static_h_n = self.gru(self.static_input, self.static_h0)
            
            self.graph_initialized = True
        except Exception:
            # If graph creation fails, reset graph state
            self.graph_initialized = False
            self.cuda_graph = None
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, 
                 otherwise (batch_size, seq_len, input_size)
        :return: h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
        """
        # Fast path: if we have a CUDA graph and input shape matches and on same device
        if (torch.cuda.is_available() and x.is_cuda and 
                self.graph_initialized and self.input_shape == x.shape and 
                self.last_device == x.device):
            # Copy input data to static tensors
            self.static_input.copy_(x)
            self.static_h0.copy_(self.h0.to(x.device))
            
            # Replay the graph
            self.cuda_graph.replay()
            
            # Return the result
            return self.static_h_n
        
        # Check if we need to initialize or reinitialize the graph
        if torch.cuda.is_available() and x.is_cuda and (not self.graph_initialized or 
                self.input_shape != x.shape or self.last_device != x.device):
            self._initialize_cuda_graph(x)
            
            # After initialization, use the graph immediately if successful
            if self.graph_initialized:
                self.static_input.copy_(x)
                self.static_h0.copy_(self.h0.to(x.device))
                self.cuda_graph.replay()
                return self.static_h_n
        
        # Standard execution fallback
        h0 = self.h0.to(x.device)
        self.gru = self.gru.to(x.device)
        
        with torch.no_grad():
            _, h_n = self.gru(x, h0)
        
        return h_n

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