import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class Model(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(Model, self).__init__()
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.graph_captured = False
    
    def forward(self, x):
        if not self.graph_captured and x.is_cuda:
            # Capture the graph on first forward pass
            self.static_input = torch.zeros_like(x)
            self.static_output = torch.zeros_like(x)
            
            # Create and capture CUDA graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = 0.5 * self.static_input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (self.static_input + 0.044715 * torch.pow(self.static_input, 3.0))))
            
            self.graph_captured = True
        
        if self.graph_captured and x.is_cuda and x.shape == self.static_input.shape:
            # Use CUDA graph
            self.static_input.copy_(x)
            self.graph.replay()
            return self.static_output.clone()
        else:
            # Fallback to regular computation
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

batch_size = 2000
dim = 2000

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []