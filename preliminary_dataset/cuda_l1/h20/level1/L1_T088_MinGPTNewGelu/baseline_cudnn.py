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
        # Default cuDNN backend flags
        self.cudnn_enabled = True
        self.cudnn_benchmark = False
        self.cudnn_deterministic = False
        self.allow_tf32 = True
    
    def forward(self, x):
        with torch.backends.cudnn.flags(enabled=self.cudnn_enabled, benchmark=self.cudnn_benchmark, deterministic=self.cudnn_deterministic, allow_tf32=self.allow_tf32):
            return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

batch_size = 2000
dim = 2000

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []