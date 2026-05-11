import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, scales the result, adds a residual connection, clamps the output,
    applies LogSumExp, and finally applies the Mish activation function.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        if self.graph is None:
            # First run: capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = x.clone()
            with torch.cuda.graph(self.graph):
                y = self.matmul(self.static_input)
                y = y * self.scale_factor
                y = y + y
                y = torch.clamp(y, self.clamp_min, self.clamp_max)
                y = torch.logsumexp(y, dim=1, keepdim=True)
                self.static_output = y * torch.nn.functional.mish(y)

        # On every run, copy input to the static buffer and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

batch_size = 128
input_size = 512
hidden_size = 1024
scale_factor = 2.0
clamp_min = -10.0
clamp_max = 10.0

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scale_factor, clamp_min, clamp_max]