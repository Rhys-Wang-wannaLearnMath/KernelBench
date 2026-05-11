import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs L2 normalization.
    """
    def __init__(self):
        """
        Initializes the L2Norm layer.

        Args:
            dim (int): Dimension along which to normalize.
        """
        super(Model, self).__init__()
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L2 normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, dim, *).

        Returns:
            torch.Tensor: Output tensor with L2 normalization applied, same shape as input.
        """
        if self.graph is None:
            # On the first run, capture the graph
            self.static_input = torch.empty_like(x)
            self.graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self.graph):
                # The model's operations are captured using a static tensor
                self.static_output = self.static_input / torch.norm(self.static_input, p=2, dim=1, keepdim=True)
        
        # For every run (including the first), copy input data and replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        
        return self.static_output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []