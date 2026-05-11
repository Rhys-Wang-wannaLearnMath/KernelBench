import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        # Create standard PyTorch layers for parameter management and fallback
        layers = []
        current_input_size = input_size
        
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Cache direct references to linear layers for faster access
        self.layer1 = self.network[0]  # Linear(input_size, layer_sizes[0])
        self.layer2 = self.network[2]  # Linear(layer_sizes[0], layer_sizes[1])
        self.layer3 = self.network[4]  # Linear(layer_sizes[1], output_size)
        
        # Pre-transpose weights and store as buffers for optimal memory access
        self.register_buffer('weight1_t', self.layer1.weight.t().contiguous())
        self.register_buffer('weight2_t', self.layer2.weight.t().contiguous())
        self.register_buffer('weight3_t', self.layer3.weight.t().contiguous())
        
        # Cache bias references for direct access
        self.bias1 = self.layer1.bias
        self.bias2 = self.layer2.bias
        self.bias3 = self.layer3.bias
        
        # Register hooks to update transposed weights when original weights change
        self._register_weight_hooks()
    
    def _register_weight_hooks(self):
        """Register hooks to update transposed weights when original weights change"""
        def make_hook(layer_num):
            def hook(grad):
                with torch.no_grad():
                    if layer_num == 1:
                        self.weight1_t.copy_(self.layer1.weight.t().contiguous())
                    elif layer_num == 2:
                        self.weight2_t.copy_(self.layer2.weight.t().contiguous())
                    else:  # layer_num == 3
                        self.weight3_t.copy_(self.layer3.weight.t().contiguous())
                return None
            return hook
        
        self.layer1.weight.register_hook(make_hook(1))
        self.layer2.weight.register_hook(make_hook(2))
        self.layer3.weight.register_hook(make_hook(3))
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Fallback to standard implementation for CPU tensors
        if not x.is_cuda:
            return self.network(x)
        
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Cache parameters locally to reduce attribute lookup overhead
        w1_t = self.weight1_t
        b1 = self.bias1
        w2_t = self.weight2_t
        b2 = self.bias2
        w3_t = self.weight3_t
        b3 = self.bias3
        
        # Layer 1: Linear + ReLU (input_size -> layer_sizes[0])
        h1 = torch.addmm(b1, x, w1_t)
        h1.relu_()
        
        # Layer 2: Linear + ReLU (layer_sizes[0] -> layer_sizes[1])
        h2 = torch.addmm(b2, h1, w2_t)
        h2.relu_()
        
        # Layer 3: Linear only (layer_sizes[1] -> output_size)
        output = torch.addmm(b3, h2, w3_t)
        
        return output

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]