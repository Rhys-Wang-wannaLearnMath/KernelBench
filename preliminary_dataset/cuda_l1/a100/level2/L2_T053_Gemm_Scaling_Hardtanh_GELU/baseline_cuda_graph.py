import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a GEMM, scaling, hardtanh, and GELU activation.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        self.gelu = nn.GELU()
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.is_capturing = False

    def forward(self, x):
        # If the flag is set, we are in the capture phase.
        # Run the original forward logic.
        if self.is_capturing:
            x = self.gemm(x)
            x = x * self.scaling_factor
            x = self.hardtanh(x)
            x = self.gelu(x)
            return x

        # On the first run, the graph is not yet captured.
        if self.graph is None:
            # Create static tensors for inputs and outputs.
            # Their shapes are determined by the first input.
            self.static_input = x.clone()

            # Create a CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Enter the graph capture context.
            with torch.cuda.graph(self.graph):
                # Set a flag to ensure the next call to forward()
                # executes the original logic.
                self.is_capturing = True
                # Run the forward pass with the static input to capture the operations.
                self.static_output = self.forward(self.static_input)
                # Reset the flag after capturing.
                self.is_capturing = False

        # For every run (including the first one after capture),
        # copy the current input's data into the static input tensor.
        self.static_input.copy_(x)

        # Replay the captured graph with the updated input.
        self.graph.replay()

        # Return a clone of the static output.
        return self.static_output.clone()

batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 0.5
hardtanh_min = -2
hardtanh_max = 2

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max]