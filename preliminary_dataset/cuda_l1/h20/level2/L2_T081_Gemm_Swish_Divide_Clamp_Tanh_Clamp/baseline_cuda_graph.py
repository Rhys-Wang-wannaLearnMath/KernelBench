import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        # CUDA Graph attributes
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        if self.graph is None:
            # First run: capture the graph.
            # We need a static input tensor with a fixed memory address.
            self.static_input = x.clone()

            # Create the graph object and capture the operations.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                y = self.gemm(self.static_input)
                y = y * torch.sigmoid(y)  # Swish activation
                y = y / 2.0
                y = torch.clamp(y, min=-1.0, max=1.0)  # Clamp between -1 and 1
                y = torch.tanh(y)  # Tanh activation
                y = torch.clamp(y, min=-1.0, max=1.0)  # Clamp between -1 and 1
                self.static_output = y
            
            # The first input `x` is already in `self.static_input` from the clone.
            # No copy is needed for the first run.
        else:
            # For subsequent runs, copy the new input data into the static buffer.
            self.static_input.copy_(x)

        # Replay the graph. For the first run, this populates the output buffer
        # for the first time. For subsequent runs, it updates it with new results.
        self.graph.replay()

        # Return a clone of the graph's output.
        return self.static_output.clone()

batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]