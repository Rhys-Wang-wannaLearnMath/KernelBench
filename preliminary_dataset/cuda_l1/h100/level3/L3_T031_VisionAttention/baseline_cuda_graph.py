import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Attention Block using Multihead Self-Attention.
        :param embed_dim: Embedding dimension (the number of channels)
        :param num_heads: Number of attention heads
        """
        super(Model, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Forward pass of the AttentionBlock.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of the same shape (B, C, H, W)
        """
        # On the first run, capture the graph.
        if self.graph is None:
            # Pre-allocate static tensors. Their memory will be reused on every run.
            self.static_input = x.clone()
            self.static_output = torch.empty_like(x)

            # Create a CUDA graph object.
            g = torch.cuda.CUDAGraph()

            # Enter graph capture context.
            with torch.cuda.graph(g):
                # Trace the model's logic using the static input tensor.
                B, C, H, W = self.static_input.shape
                x_graph = self.static_input.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
                attn_output, _ = self.attn(x_graph, x_graph, x_graph)
                x_graph = self.norm(attn_output + x_graph)  # (seq_len, batch_size, embed_dim)
                result = x_graph.permute(1, 2, 0).view(B, C, H, W)

                # The captured graph will write its result into the pre-allocated static_output tensor.
                self.static_output.copy_(result)

            # Save the captured graph.
            self.graph = g

        # For every run (including the first one after capture),
        # copy the current input data to the static input tensor.
        self.static_input.copy_(x)

        # Replay the captured graph. This executes the traced operations on the GPU.
        self.graph.replay()

        # Return the static output tensor, which now contains the result of the replay.
        return self.static_output

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.randn(batch_size, num_channels, image_height, image_width)]

def get_init_inputs():
    return [embed_dim, num_heads]