import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a HardSigmoid activation.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Initialize attributes for CUDA graph. They will be populated on the first forward pass.
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        # If the graph has not been captured yet, record it.
        if self.graph is None:
            # Create static tensors that will serve as memory buffers for the graph.
            self.static_input = x.clone()

            # Create a new CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Begin capturing the operations into the graph.
            with torch.cuda.graph(self.graph):
                # Run the model's operations using the static input tensor.
                # The resulting operations are recorded into the graph.
                self.static_output = torch.nn.functional.hardsigmoid(self.static_input)
        
        # Copy the current input data into the static input buffer used by the graph.
        self.static_input.copy_(x)
        
        # Replay the captured graph. This executes the recorded CUDA kernels
        # with minimal CPU overhead.
        self.graph.replay()
        
        # Return the output from the static output buffer.
        return self.static_output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed