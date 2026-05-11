import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a LeakyReLU activation.
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initializes the LeakyReLU module.

        Args:
            negative_slope (float, optional): The negative slope of the activation function. Defaults to 0.01.
        """
        super(Model, self).__init__()
        self.negative_slope = negative_slope
        
        # Placeholders for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LeakyReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with LeakyReLU applied, same shape as input.
        """
        if self.graph is None:
            # On the first run, capture the graph
            self.static_input = x.clone()
            self.graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self.graph):
                self.static_output = torch.nn.functional.leaky_relu(
                    self.static_input, negative_slope=self.negative_slope
                )

        # Copy the current input to the graph's static input tensor
        self.static_input.copy_(x)
        
        # Replay the captured graph
        self.graph.replay()
        
        # Return a clone of the graph's static output
        return self.static_output.clone()

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed