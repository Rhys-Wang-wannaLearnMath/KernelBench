import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(Model, self).__init__()
        self.graph = None
        self.static_A = None
        self.static_B = None
        self.static_C = None
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        if self.graph is None:
            # Create static tensors for inputs. These will be populated with new data
            # on each forward pass.
            self.static_A = torch.empty_like(A)
            self.static_B = torch.empty_like(B)
            
            # Instantiate the graph
            self.graph = torch.cuda.CUDAGraph()
            
            # Begin capturing the graph
            with torch.cuda.graph(self.graph):
                # The operations to be captured
                self.static_C = torch.diag(self.static_A) @ self.static_B

        # Copy the current input data to the static tensors
        self.static_A.copy_(A)
        self.static_B.copy_(B)
        
        # Replay the captured graph
        self.graph.replay()
        
        # Return the output from the static output tensor
        return self.static_C

M = 4096
N = 4096

def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed