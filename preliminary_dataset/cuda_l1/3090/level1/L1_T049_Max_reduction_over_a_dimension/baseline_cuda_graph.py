import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        # On the first forward pass, the graph is not yet captured.
        # We need to capture it.
        if self.graph is None:
            # Create the graph object
            self.graph = torch.cuda.CUDAGraph()

            # The static tensors must be created on the same device as the input
            # and have the same shape.
            self.static_input = torch.empty_like(x)
            
            # Use the context manager to capture the graph.
            # All operations inside the context are recorded.
            with torch.cuda.graph(self.graph):
                # The operations within the graph must use the static tensors
                static_y = torch.max(self.static_input, dim=self.dim)[0]
                # Store the static output tensor to access the result after replay
                self.static_output = static_y
        
        # For every call (including the first), copy the current input's data
        # into the static input tensor used by the graph.
        self.static_input.copy_(x)
        
        # Replay the captured graph. The operations are executed on the GPU,
        # and the result is placed in self.static_output.
        self.graph.replay()
        
        # Return the static output tensor.
        return self.static_output

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1] # Example, change to desired dimension