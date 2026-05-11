import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies ReLU, and divides by a constant.
    """
    def __init__(self, in_features, out_features, divisor):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # If the graph has not been captured yet
        if self.graph is None:
            # Create a new CUDA graph
            self.graph = torch.cuda.CUDAGraph()
            # Assign the input tensor to a static attribute for capture
            self.static_input = x

            # Begin capturing the graph
            with torch.cuda.graph(self.graph):
                # Define the model's operations using the static input
                # The result is stored in the static output attribute
                y = self.linear(self.static_input)
                y = torch.relu(y)
                self.static_output = y / self.divisor
            
            # For the first run, execute the captured graph to populate the output
            self.graph.replay()
        else:
            # For subsequent runs, copy new input data to the static input tensor
            self.static_input.copy_(x)
            # Replay the captured graph with the new input data
            self.graph.replay()

        # Return a clone of the static output tensor
        return self.static_output.clone()

batch_size = 128
in_features = 1024
out_features = 512
divisor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, divisor]