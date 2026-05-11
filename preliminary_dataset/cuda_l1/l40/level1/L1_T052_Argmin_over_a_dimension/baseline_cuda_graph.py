import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmin on.

        Args:
            dim (int): Dimension along which to find the minimum value.
        """
        super(Model, self).__init__()
        self.dim = dim
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        if self.graph is None:
            # First run: capture the graph.
            # The input tensor `x` is saved as a static attribute. Its memory
            # will be used for all subsequent runs.
            self.static_input = x
            
            # Create a new CUDA graph.
            self.graph = torch.cuda.CUDAGraph()
            
            # Enter graph capture context. Operations are recorded but not run.
            with torch.cuda.graph(self.graph):
                # The model's operations are recorded using the static tensors.
                self.static_output = torch.argmin(self.static_input, dim=self.dim)
            
            # For the first run, the input data is already in self.static_input.
            # We fall through to the replay section to execute the graph.
        else:
            # Subsequent runs: copy the new input data into the static tensor.
            self.static_input.copy_(x)
            
        # Replay the captured graph. This executes the recorded operations.
        # On the first run, this populates the static output.
        # On subsequent runs, this recomputes the output with new data.
        self.graph.replay()
        
        return self.static_output

batch_size = 16
dim1 = 256
dim2 = 256
dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [dim]