import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar, and applies GELU activation.
    """
    def __init__(self, input_size, output_size, divisor):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.divisor = divisor
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        if self.graph is None:
            # On the first run, capture the graph
            self.static_input = torch.empty_like(x)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                # Place the model's operations inside the capture block
                y = self.linear(self.static_input)
                y = y / self.divisor
                y = torch.nn.functional.gelu(y)
                self.static_output = y
            self.graph = g

        # For every run (including the first), copy the new data and replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

batch_size = 128
input_size = 512
output_size = 1024
divisor = 10.0

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, output_size, divisor]