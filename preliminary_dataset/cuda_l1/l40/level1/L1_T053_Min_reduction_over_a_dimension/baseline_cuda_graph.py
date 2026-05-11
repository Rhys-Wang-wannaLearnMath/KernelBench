import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs min reduction over a specific dimension.
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
        Applies min reduction over the specified dimension to the input tensor.
        This implementation uses CUDA graphs for performance improvement in inference.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        # In training mode, fall back to eager execution
        if self.training:
            return torch.min(x, dim=self.dim)[0]

        # If the graph is not captured yet, capture it
        if self.graph is None:
            # Create static tensors. These are placeholders for graph capture.
            self.static_input = torch.empty_like(x)

            # Begin graph capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                # Run the model's logic with static tensors
                self.static_output = torch.min(self.static_input, dim=self.dim)[0]
            self.graph = g

        # Copy input data to the static input tensor
        self.static_input.copy_(x)

        # Replay the graph
        self.graph.replay()

        # Return the static output tensor which now contains the result
        return self.static_output

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1] # Example, change to desired dimension