import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # Create a static input tensor with the same properties as the input 'x'.
            # This tensor will be used as a fixed memory buffer for all subsequent inputs.
            self.static_input = torch.empty_like(x)
            
            # Create the CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()
            
            # Enter graph capture mode.
            with torch.cuda.graph(self.graph):
                # The model's operations are defined once using the static input tensor.
                y = self.gemm(self.static_input)
                y = y * self.scale
                # The result is assigned to a static output tensor.
                self.static_output = self.bn(y)
        
        # Copy the data from the current input 'x' to the static input buffer.
        self.static_input.copy_(x)
        
        # Replay the captured graph. The operations are executed on the GPU,
        # and the result is placed into self.static_output.
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