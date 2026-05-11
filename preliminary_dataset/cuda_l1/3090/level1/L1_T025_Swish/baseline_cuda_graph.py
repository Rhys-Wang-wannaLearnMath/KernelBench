import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Swish activation.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Attributes for CUDA graph functionality
        self.graph = None
        self.static_input = None
        self.static_output = None
        # A dedicated stream is required for CUDA graph capture
        self.stream = torch.cuda.Stream()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        # On the first call, capture the CUDA graph.
        if self.graph is None:
            # Graph capture must be done on a non-default stream.
            # We sync the new stream with the default stream to ensure the input 'x' is ready.
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                # Create a static tensor that will be used for graph replay.
                self.static_input = x.clone()
                
                # Begin capture on the non-default stream.
                self.graph = torch.cuda.CUDAGraph()
                self.graph.capture_begin()
                
                # Run the model's operations using the static tensors.
                # This sequence of operations is recorded in the graph.
                self.static_output = self.static_input * torch.sigmoid(self.static_input)

                # End capture.
                self.graph.capture_end()
            
            # Sync the default stream with the new stream to ensure capture is complete.
            torch.cuda.current_stream().wait_stream(self.stream)

        # For every run (including the first), copy the current input data
        # to the static input tensor.
        self.static_input.copy_(x)
        
        # Replay the captured graph. This executes the recorded operations
        # on the static tensors.
        self.graph.replay()
        
        # Return a clone of the static output tensor to avoid memory corruption.
        return self.static_output.clone()

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed