import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        super(Model, self).__init__()
        # Attributes for CUDA graph
        self.graph = None
        self.static_A = None
        self.static_B = None
        self.static_C = None
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        if self.graph is None:
            # On the first forward pass, we record the CUDA graph.
            # 1. Create static tensors for inputs and outputs. These have fixed
            #    memory addresses, which is a requirement for CUDA graph capture.
            self.static_A = A.clone()
            self.static_B = B.clone()
            # The output tensor must also be pre-allocated. We run the operation
            # once to determine the correct output shape and device.
            self.static_C = torch.empty_like(torch.matmul(A, B))

            # 2. Instantiate and capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # The captured operations must use the static tensors and write
                # to the static output tensor using the 'out' argument.
                torch.matmul(self.static_A, self.static_B, out=self.static_C)

        # For every forward pass (including the first one after capture),
        # 1. Copy the current input data into the static input tensors.
        self.static_A.copy_(A)
        self.static_B.copy_(B)
        
        # 2. Replay the graph. This executes the captured kernels with the updated
        #    input data, bypassing the Python interpreter for significant speedup.
        self.graph.replay()

        # 3. Return a clone of the static output tensor. This is crucial for
        #    correctness, as it returns a tensor with the correct data for this
        #    pass without exposing the graph's internal static tensor to the caller.
        return self.static_C.clone()

M = 16384
N = 16

def get_inputs():
    A = torch.randn(M, N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed