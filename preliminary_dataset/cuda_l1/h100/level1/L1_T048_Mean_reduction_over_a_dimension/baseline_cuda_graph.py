import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs mean reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean.
        This method uses CUDA graphs for acceleration. The graph is captured on the
        first call and replayed on subsequent calls.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension. The shape of the output is the same as the input except for the reduced dimension which is removed.
        """
        if self.graph is None:
            # On the first pass, capture the graph
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = x.clone()

            with torch.cuda.graph(self.graph):
                self.static_output = torch.mean(self.static_input, dim=self.dim)

        # Copy the current input to the static memory region used by the graph
        self.static_input.copy_(x)
        
        # Replay the captured graph
        self.graph.replay()
        
        # Return a clone of the output tensor
        return self.static_output.clone()

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]