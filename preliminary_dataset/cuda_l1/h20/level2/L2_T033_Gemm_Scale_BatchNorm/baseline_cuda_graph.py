import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a GEMM (general matrix multiplication), applies scaling, 
    and then batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        # A persistent stream is required for graph capture
        self.stream = torch.cuda.Stream()

    def forward(self, x):
        # On the first forward pass, we capture the model's operations into a graph.
        if self.graph is None:
            # The static input tensor must be on the same device as the model and input.
            self.static_input = x.clone()

            # Begin capturing the graph on a non-default stream.
            with torch.cuda.stream(self.stream):
                self.graph = torch.cuda.CUDAGraph()
                self.graph.capture_begin()

                # Run the forward pass with the static input. These operations are recorded.
                y = self.gemm(self.static_input)
                y = y * self.scale
                # The final result of the captured pass is assigned to the static output tensor.
                self.static_output = self.bn(y)
                
                self.graph.capture_end()

            # Ensure the capture on the side stream is complete before proceeding.
            torch.cuda.current_stream().wait_stream(self.stream)

        # For all passes (including the first), copy the current input data to the static input tensor.
        self.static_input.copy_(x)

        # Replay the captured graph. The operations run on the GPU, and the result
        # is placed in self.static_output.
        self.graph.replay()

        # Return a clone of the static output tensor.
        return self.static_output.clone()

batch_size = 128
in_features = 1024
out_features = 512
scale_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scale_shape]