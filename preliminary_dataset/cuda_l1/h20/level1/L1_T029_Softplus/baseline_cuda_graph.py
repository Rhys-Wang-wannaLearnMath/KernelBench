import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Softplus activation.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        # On the first forward pass, the graph is not yet captured.
        if self.graph is None:
            # Create static tensors for input and output. These will be used
            # as placeholders for graph capture and replay.
            self.static_input = torch.empty_like(x)
            self.static_output = torch.empty_like(x)
            
            # Instantiate the graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Begin graph capture.
            with torch.cuda.graph(self.graph):
                # The model's operations are defined here, using the static tensors.
                # For non-in-place operations, the result must be copied into
                # the static output tensor.
                graph_run_output = torch.nn.functional.softplus(self.static_input)
                self.static_output.copy_(graph_run_output)
        
        # Copy the current input data into the static input tensor.
        self.static_input.copy_(x)
        
        # Replay the captured graph. This executes the defined operations
        # with the current data in self.static_input and places the
        # result in self.static_output.
        self.graph.replay()
        
        # Return the output from the static output tensor.
        return self.static_output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed