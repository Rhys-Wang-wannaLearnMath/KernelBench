import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(Model, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        
        # CUDA graph attributes
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.graph_initialized = False
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, otherwise (batch_size, seq_len, input_size)
        :param h_0: The initial hidden state for the input sequence, shape (num_layers * num_directions, batch_size, hidden_size) (default: None)
        :return: output, h_n
            - output: The output features (h_t) from the last layer of the GRU, for each t, shape (seq_len, batch_size, num_directions * hidden_size) if batch_first=False, otherwise (batch_size, seq_len, num_directions * hidden_size)
            - h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
        """
        if x.is_cuda and not self.graph_initialized:
            # Initialize CUDA graph on first CUDA execution
            self._setup_cuda_graph(x)
        
        if x.is_cuda and self.graph is not None:
            # Use CUDA graph execution
            self.static_input.copy_(x)
            self.graph.replay()
            return self.static_output.clone()
        else:
            # Fallback to regular execution
            self.h0 = self.h0.to(x.device)
            output, h_n = self.gru(x, self.h0)
            return output
    
    def _setup_cuda_graph(self, x):
        """Setup CUDA graph for the forward pass"""
        try:
            # Ensure h0 is on the correct device before graph capture
            self.h0 = self.h0.to(x.device)
            
            # Create static tensors
            self.static_input = torch.zeros_like(x)
            
            # Ensure all operations are synchronized before capturing
            torch.cuda.synchronize()
            
            # Capture the graph
            self.graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self.graph):
                output, h_n = self.gru(self.static_input, self.h0)
                self.static_output = output
            
            self.graph_initialized = True
        except Exception:
            # If graph capture fails, disable graph execution
            self.graph = None
            self.graph_initialized = True

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