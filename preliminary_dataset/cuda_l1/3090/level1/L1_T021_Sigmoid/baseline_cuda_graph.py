import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Sigmoid activation.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Initialize attributes for CUDA graph caching.
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
        """
        if self.graph is None:
            # First forward pass:
            # 1. Compute the output eagerly to ensure the first result is correct.
            # 2. Set up and capture the graph for subsequent runs.
            
            # 1. Eagerly compute the first output.
            eager_output = torch.sigmoid(x)

            # 2. Set up for graph capture.
            # Create static tensors with the same properties as the first input/output.
            # These tensors will be used as placeholders for all subsequent replays.
            self.static_input = x.clone()
            self.static_output = eager_output.clone()
            
            # Create a new CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Begin capturing the graph. All operations within this context are recorded.
            with torch.cuda.graph(self.graph):
                # The model's operations are defined here using the static tensors.
                # The `out` argument ensures the result is written into the pre-allocated
                # static output tensor, avoiding memory allocation during replay.
                torch.sigmoid(self.static_input, out=self.static_output)
            
            # Return the eagerly computed output for the first run.
            return eager_output
        else:
            # Subsequent forward passes: replay the captured graph.
            # 1. Copy the new input data into the static input tensor.
            self.static_input.copy_(x)
            
            # 2. Replay the graph. This executes the captured CUDA kernels.
            self.graph.replay()
            
            # 3. Return a clone of the static output. Cloning ensures that a new
            # tensor is returned, matching the original model's behavior and
            # preventing side-effects.
            return self.static_output.clone()

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed