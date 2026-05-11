import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs matrix multiplication, max pooling, sum, and scaling.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.max_pool = nn.MaxPool1d(kernel_size)
        self.scale_factor = scale_factor
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
            # On the first forward pass, capture the model's operations into a CUDA graph.
            self.static_input = x.clone()

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # The original forward logic is placed inside the graph capture context.
                # It operates on the static input tensor.
                static_y = self.matmul(self.static_input)
                static_y = self.max_pool(static_y.unsqueeze(1)).squeeze(1)
                static_y = torch.sum(static_y, dim=1)
                self.static_output = static_y * self.scale_factor

        # For every forward pass, copy the new input data to the static input tensor.
        self.static_input.copy_(x)

        # Replay the captured graph.
        self.graph.replay()

        # Return the static output tensor, which is updated in-place by the graph replay.
        return self.static_output

batch_size = 128
in_features = 10
out_features = 5
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]