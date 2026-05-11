import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim
        self.graph = None
        self.static_inputs = None
        self.static_output = None

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        if self.graph is None:
            # First run: capture the graph.
            # We need static tensors that persist across calls, so we clone the first inputs.
            self.static_inputs = (x.clone(), mask.clone())
            
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # Define the graph's operations using the static tensors.
                self.static_output = torch.cumsum(self.static_inputs[0] * self.static_inputs[1], dim=self.dim)

        # For every run (including the first), copy the current input data into our
        # static tensors and replay the graph.
        self.static_inputs[0].copy_(x)
        self.static_inputs[1].copy_(mask)
        self.graph.replay()
        return self.static_output

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    x = torch.randn(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()  # Random boolean mask
    return [x, mask]

def get_init_inputs():
    return [dim]