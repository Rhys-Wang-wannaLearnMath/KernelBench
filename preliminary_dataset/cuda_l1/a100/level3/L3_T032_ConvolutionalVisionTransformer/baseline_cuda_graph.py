import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, 
                 mlp_ratio=4.0, patch_size=4, in_channels=3):
        """
        Convolutional Vision Transformer (CViT) implementation.
        :param num_classes: Number of output classes for classification.
        :param embed_dim: Dimensionality of the embedding space.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of transformer layers.
        :param mlp_ratio: Ratio of the MLP hidden dimension to the embedding dimension.
        :param patch_size: Size of the convolutional patches.
        :param in_channels: Number of input channels (e.g., 3 for RGB images).
        """
        super(Model, self).__init__()

        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten()
        
        # Linear projection to create embeddings
        self.linear_proj = nn.Linear(embed_dim * (32 // patch_size) * (32 // patch_size), embed_dim)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                        dim_feedforward=int(embed_dim * mlp_ratio), dropout=0.0)
            for _ in range(num_layers)
        ])
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fc_out = nn.Linear(embed_dim, num_classes)
        
        # CUDA Graph initialization
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Forward pass of the CViT model.
        :param x: Input tensor of shape (B, C, H, W)
        :return: Output tensor of shape (B, num_classes)
        """
        if self.training or self.graph is None:
            if not self.training and self.graph is None and x.is_cuda:
                self._initialize_cuda_graph(x)
            return self._forward_impl(x)
        
        # Use CUDA graph for inference
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

    def _forward_impl(self, x):
        """
        Actual forward implementation.
        """
        B, C, H, W = x.shape
        
        x = self.conv1(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = self.flatten(x)  # (B, embed_dim * (H/patch_size) * (W/patch_size))
        x = self.linear_proj(x)  # (B, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)  # (B, 1+N, embed_dim)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Classify based on cls token
        x = x[:, 0]  # Get the cls token's output
        x = self.fc_out(x)  # (B, num_classes)
        
        return x

    def _initialize_cuda_graph(self, x):
        """
        Initialize CUDA graph for inference.
        """
        torch.cuda.synchronize()
        
        # Create static tensors
        self.static_input = x.clone()
        
        # Get output shape
        with torch.no_grad():
            sample_output = self._forward_impl(x)
            self.static_output = torch.empty_like(sample_output)
        
        # Capture the graph
        self.graph = torch.cuda.CUDAGraph()
        
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self.static_output = self._forward_impl(self.static_input)
        torch.cuda.synchronize()
    
batch_size = 10
image_size = 32
embed_dim = 128
in_channels = 3
num_heads = 4
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, in_channels, image_size, image_size)]

def get_init_inputs():
    return [num_classes, embed_dim, num_heads]