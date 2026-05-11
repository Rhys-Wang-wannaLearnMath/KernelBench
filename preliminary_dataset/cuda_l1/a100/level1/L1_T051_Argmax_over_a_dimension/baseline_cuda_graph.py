import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Argmax over a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(Model, self).__init__()
        self.dim = dim
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        if self.graph is None:
            # On the first run, capture the CUDA graph
            self.static_input = torch.empty_like(x)
            self.graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self.graph):
                self.static_output = torch.argmax(self.static_input, dim=self.dim)

        # Copy the current input to the static placeholder
        self.static_input.copy_(x)
        # Replay the captured graph
        self.graph.replay()
        # Return a clone of the static output
        return self.static_output.clone()

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]