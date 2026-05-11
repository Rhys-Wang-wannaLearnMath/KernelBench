import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies minimum, and subtracts a constant.
    """
    def __init__(self, in_features, out_features, constant):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.constant = nn.Parameter(torch.tensor(constant))
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # CUDA graphs are only for CUDA tensors
        if not x.is_cuda:
            x = self.linear(x)
            x = torch.min(x, self.constant)
            x = x - self.constant
            return x

        # On the first run, capture the graph
        if self.graph is None:
            # Create persistent input and output tensors
            self.static_input = x.clone()
            
            # Instantiate the graph
            self.graph = torch.cuda.CUDAGraph()

            # Begin capture
            with torch.cuda.graph(self.graph):
                # Run the model's operations with the static input
                y = self.linear(self.static_input)
                y = torch.min(y, self.constant)
                # Assign the final result to the static output tensor
                self.static_output = y - self.constant

        # Copy the current input's data to the static input tensor
        self.static_input.copy_(x)
        
        # Replay the graph
        self.graph.replay()
        
        # Return a clone of the static output
        return self.static_output.clone()

batch_size = 128
in_features = 10
out_features = 5
constant = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, constant]