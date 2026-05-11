import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1, emb_dropout=0.1):
        """
        Vision Transformer (ViT) model.

        :param image_size: The size of the input image (assumed to be square).
        :param patch_size: The size of each patch (assumed to be square).
        :param num_classes: The number of output classes.
        :param dim: The dimensionality of the embedding space.
        :param depth: The number of transformer layers.
        :param heads: The number of attention heads.
        :param mlp_dim: The dimensionality of the MLP (Multi-Layer Perceptron) in the transformer.
        :param channels: The number of channels in the input image (default is 3 for RGB).
        :param dropout: Dropout rate applied in the MLP.
        :param emb_dropout: Dropout rate applied to the embedded patches.
        """
        super(Model, self).__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )
        
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, img):
        """
        Forward pass of the Vision Transformer.

        :param img: The input image tensor, shape (batch_size, channels, image_size, image_size).
        :return: The output tensor, shape (batch_size, num_classes).
        """
        if not img.is_cuda:
            # Fallback to eager mode for CPU execution
            p = self.patch_size
            
            x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, p*p*img.shape[1])
            x = self.patch_to_embedding(x)
            
            cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding
            x = self.dropout(x)
            
            x = self.transformer(x)
            
            x = self.to_cls_token(x[:, 0])
            return self.mlp_head(x)

        # On the first run or if input shape changes, capture the graph
        if self.graph is None or self.static_input.shape != img.shape:
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = img.clone()

            with torch.cuda.graph(self.graph):
                p = self.patch_size
        
                x = self.static_input.unfold(2, p, p).unfold(3, p, p).reshape(self.static_input.shape[0], -1, p*p*self.static_input.shape[1])
                x = self.patch_to_embedding(x)
                
                cls_tokens = self.cls_token.expand(self.static_input.shape[0], -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x += self.pos_embedding
                x = self.dropout(x)
                
                x = self.transformer(x)
                
                x = self.to_cls_token(x[:, 0])
                self.static_output = self.mlp_head(x)
        
        # Copy the current input to the static buffer and replay the graph
        self.static_input.copy_(img)
        self.graph.replay()
        return self.static_output.clone()

# Test code
image_size = 224
patch_size = 16
num_classes = 10
dim = 512
depth = 6
heads = 8
mlp_dim = 2048
channels = 3
dropout = 0.0
emb_dropout = 0.0

def get_inputs():
    return [torch.randn(2, channels, image_size, image_size)]

def get_init_inputs():
    return [image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout]