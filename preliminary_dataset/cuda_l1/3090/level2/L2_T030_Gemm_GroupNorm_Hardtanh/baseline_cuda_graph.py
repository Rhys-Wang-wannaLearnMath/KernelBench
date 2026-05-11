import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a GEMM, applies Group Normalization, and then HardTanh.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        # A non-default stream is required for graph capture
        self.stream = torch.cuda.Stream()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # On the first run, capture the graph.
        if self.graph is None:
            # Create a static input tensor. It will be populated with real data before each run.
            self.static_input = torch.empty_like(x)

            # Graph capture must be done on a non-default stream.
            with torch.cuda.stream(self.stream):
                self.graph = torch.cuda.CUDAGraph()
                self.graph.capture_begin()

                # The actual model operations using the static input tensor.
                y = self.gemm(self.static_input)
                y = self.group_norm(y)
                self.static_output = self.hardtanh(y)

                self.graph.capture_end()

            # Wait for the graph capture to complete.
            torch.cuda.synchronize()

        # For every run, copy the input data to the static input tensor and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output

batch_size = 128
in_features = 1024
out_features = 512
num_groups = 8
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]