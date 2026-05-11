import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
        # Attributes for CUDA graph
        self.graph = None
        self.static_A = None
        self.static_B = None
        self.static_C = None
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        # On the first forward pass, capture the CUDA graph
        if self.graph is None:
            # Create static tensors. These are memory buffers that will be
            # used for all subsequent graph replays.
            self.static_A = torch.empty_like(A)
            self.static_B = torch.empty_like(B)
            
            # Instantiate the graph object
            self.graph = torch.cuda.CUDAGraph()

            # Begin graph capture. All operations within this context are recorded.
            with torch.cuda.graph(self.graph):
                # The computation is defined using the static tensors
                self.static_C = torch.matmul(self.static_A, self.static_B)

        # Copy the current input data to the static tensors
        self.static_A.copy_(A)
        self.static_B.copy_(B)
        
        # Replay the captured graph. This executes the recorded operations
        # on the data that was just copied.
        self.graph.replay()
        
        # Return the output from the static output tensor
        return self.static_C

N = 2048

def get_inputs():
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed