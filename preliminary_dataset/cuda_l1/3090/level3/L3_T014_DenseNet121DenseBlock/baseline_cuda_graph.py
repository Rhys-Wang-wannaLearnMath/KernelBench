import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(Model, self).__init__()
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.stream = torch.cuda.Stream()
        
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )
    
    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        if self.graph is None:
            # On the first run, capture the graph on a non-default stream.
            with torch.cuda.stream(self.stream):
                self.static_input = x.clone()
                self.graph = torch.cuda.CUDAGraph()
                
                self.graph.capture_begin()
                
                # --- Original forward logic using the static input tensor ---
                features = [self.static_input]
                graphed_x = self.static_input
                for layer in self.layers:
                    new_feature = layer(graphed_x)
                    features.append(new_feature)
                    graphed_x = torch.cat(features, 1)
                # --- End of original forward logic ---
                
                self.static_output = graphed_x
                self.graph.capture_end()

            # Wait for the capture to complete on the non-default stream before replaying.
            torch.cuda.current_stream().wait_stream(self.stream)
        
        # Copy the current input data to the static buffer.
        self.static_input.copy_(x)
        
        # Replay the captured graph.
        self.graph.replay()
        
        # Return a clone of the graph's output tensor.
        return self.static_output.clone()
    
batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_layers, num_input_features , growth_rate]