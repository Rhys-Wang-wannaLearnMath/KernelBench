import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model implementation for benchmarking.
        
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
        super(Model, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

        # CUDA graph attributes
        self.graph = None
        self.static_X = None
        self.static_initial_states = None
        self.static_Y = None
        
    def segsum(self, x):
        """Naive segment sum calculation."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
    
    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Output tensor Y and final state
        """
        # If graph is not captured, this is the first run.
        if self.graph is None:
            # Make control flow static by always having a tensor for initial_states.
            # If the user provides None, we create a zero tensor.
            _initial_states = initial_states
            if _initial_states is None:
                # The shape of initial_states should be (batch, 1, n_heads, d_head, d_state)
                initial_states_shape = (self.batch_size, 1, self.n_heads, self.d_head, self.d_state)
                _initial_states = torch.zeros(initial_states_shape, device=X.device, dtype=X.dtype)

            # Assign static tensors that will be used for capture and replay.
            # These tensors hold the memory buffers.
            self.static_X = X
            self.static_initial_states = _initial_states

            # Capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # --- Start of original forward logic, using static inputs ---
                
                # Rearrange into blocks/chunks
                X_blocks, A_blocks, B_blocks, C_blocks = [
                    rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
                    for x in (self.static_X, self.A, self.B, self.C)
                ]
                
                A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
                A_cumsum = torch.cumsum(A_blocks, dim=-1)
                
                # 1. Compute diagonal block outputs
                L = torch.exp(self.segsum(A_blocks))
                Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                                     C_blocks, B_blocks, L, X_blocks)
                
                # 2. Compute intra-chunk states
                decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
                states = torch.einsum("bclhn,bhcl,bclhp->bchpn", 
                                    B_blocks, decay_states, X_blocks)
                
                # 3. Compute inter-chunk recurrence
                # The control flow `if initial_states is None:` is removed from the captured path.
                # We now always have a tensor in self.static_initial_states.
                states = torch.cat([self.static_initial_states, states], dim=1)
                
                decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
                new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
                states = new_states[:, :-1]
                
                # 4. Compute state-to-output conversion
                state_decay_out = torch.exp(A_cumsum)
                Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', 
                                   C_blocks, states, state_decay_out)
                
                # Combine diagonal and off-diagonal terms
                Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

                # --- End of original forward logic ---
                
                # Store the output tensor of the graph
                self.static_Y = Y

            # Perform a replay to populate the output tensor for the first run
            self.graph.replay()
            return self.static_Y

        else:
            # Graph has been captured, replay it.
            # Copy the new input data into the static tensors' memory.
            self.static_X.copy_(X)
            if initial_states is not None:
                self.static_initial_states.copy_(initial_states)
            else:
                # If no initial state is provided, use zeros, matching the original logic.
                self.static_initial_states.zero_()

            self.graph.replay()
            return self.static_Y

# Test parameters
batch_size = 16
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    return [torch.randn(batch_size, seq_length, n_heads, d_head)]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]