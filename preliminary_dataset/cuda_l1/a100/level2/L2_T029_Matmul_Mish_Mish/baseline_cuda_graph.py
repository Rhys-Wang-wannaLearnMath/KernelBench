import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # CUDA graph attributes
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first forward pass, capture the model's operations into a graph.
        if self.graph is None:
            # The input tensor for the first run is used as a static placeholder
            # for the graph's input.
            self.static_input = x
            
            # Create a new graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Begin capturing the operations into the graph.
            with torch.cuda.graph(self.graph):
                # The original forward logic is placed here.
                # It is traced using the static input tensor.
                y = self.linear(self.static_input)
                y = torch.nn.functional.mish(y)
                self.static_output = torch.nn.functional.mish(y)

        # For every run (including the first), copy the current input's data
        # into the static placeholder and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        
        # Return the static output tensor, which has been updated by the replay.
        return self.static_output

batch_size = 128
in_features = 10
out_features = 20

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]