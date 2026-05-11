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

        # CUDA Graph attributes
        self.graph = None
        self.static_input = None
        self.static_h0 = None
        self.static_output_h_n = None
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, otherwise (batch_size, seq_len, input_size)
        :param h_0: The initial hidden state for the input sequence, shape (num_layers * num_directions, batch_size, hidden_size) (default: None)
        :return: output, h_n
            - output: The output features (h_t) from the last layer of the GRU, for each t, shape (seq_len, batch_size, num_directions * hidden_size) if batch_first=False, otherwise (batch_size, seq_len, num_directions * hidden_size)
            - h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
        """
        if self.graph is None:
            # First call, capture the graph.
            # Move the model and the initial hidden state to the input's device.
            self.to(x.device)
            self.h0 = self.h0.to(x.device)

            # Create static placeholders for inputs.
            self.static_input = x.clone()
            self.static_h0 = self.h0.clone()

            # Instantiate and capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                _, static_h_n = self.gru(self.static_input, self.static_h0)
            
            # The output of the captured region is also a static tensor.
            # We save it to be populated during replay.
            self.static_output_h_n = static_h_n

        # For every call, copy the current input data to the static placeholder.
        self.static_input.copy_(x)
        
        # Replay the graph.
        self.graph.replay()
        
        # Return a clone of the static output.
        return self.static_output_h_n.clone()

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