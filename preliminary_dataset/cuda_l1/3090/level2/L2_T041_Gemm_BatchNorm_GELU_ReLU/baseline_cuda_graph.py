import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a GEMM, BatchNorm, GELU, GroupNorm, Mean, and ReLU operations in sequence.
    """
    def __init__(self, in_features, out_features, num_groups):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        
        # Attributes for CUDA graph
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
            # Create the graph instance
            g = torch.cuda.CUDAGraph()

            # Define static tensors for inputs and outputs.
            # The output shape is calculated manually to avoid a warmup run.
            self.static_input = torch.empty_like(x)
            output_shape = (x.shape[0], 1)
            self.static_output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

            # Capture the graph
            with torch.cuda.graph(g):
                y = self.gemm(self.static_input)
                y = self.batch_norm(y)
                y = torch.nn.functional.gelu(y)
                y = self.group_norm(y)
                y = torch.mean(y, dim=1, keepdim=True)
                y = torch.relu(y)
                # The result must be copied into the pre-allocated static output tensor
                self.static_output.copy_(y)

            # Save the graph for future replays
            self.graph = g

        # For every forward pass, copy the new data to the static input tensor
        # and replay the captured graph.
        self.static_input.copy_(x)
        self.graph.replay()

        # Return a clone of the output tensor to avoid memory aliasing issues
        return self.static_output.clone()

batch_size = 128
in_features = 512
out_features = 1024
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups]