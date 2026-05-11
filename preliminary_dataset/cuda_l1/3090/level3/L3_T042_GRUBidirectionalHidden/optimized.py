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
        
        # Register the initial hidden state as a buffer to ensure it's properly moved with the module
        self.register_buffer('h0', torch.randn((num_layers * 2, batch_size, hidden_size)))
        
        # Store parameters for later use
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # CUDA graph optimization state (removed the cache mechanism)
        
        # Try to use torch.compile if available (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                self.compiled_gru = torch.compile(self.gru)
                self.has_compile = True
            else:
                self.has_compile = False
        except:
            self.has_compile = False
    
    # Removed _get_cache_key method
    
    def _ensure_device_consistency(self, x):
        """Ensure all tensors are on the same device"""
        if self.h0.device != x.device:
            self.h0 = self.h0.to(x.device)
        
        if next(self.gru.parameters()).device != x.device:
            self.gru = self.gru.to(x.device)
    
    # Removed _initialize_cuda_graph method
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, otherwise (batch_size, seq_len, input_size)
        :return: h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
        """
        # Ensure model is on the same device as input
        self._ensure_device_consistency(x)
        
        # Removed the cache logic; proceed with regular execution
        with torch.no_grad():
            if self.has_compile and x.is_cuda:
                _, h_n = self.compiled_gru(x, self.h0)
            else:
                _, h_n = self.gru(x, self.h0)
        
        return h_n

# Test code with the same hyperparameters as the reference implementation
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.randn(seq_len, batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]