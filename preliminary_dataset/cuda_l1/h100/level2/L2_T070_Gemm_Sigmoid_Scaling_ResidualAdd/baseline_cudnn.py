import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        self.cudnn_flags = None

    def set_cudnn_flags(self, flags):
        """
        Set cuDNN backend flags for the forward pass.

        Args:
            flags (dict): A dictionary of cuDNN flags to enable.
                          e.g., {'benchmark': True, 'deterministic': False}
        """
        self.cudnn_flags = flags

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        def _forward_impl(input_tensor):
            output = self.gemm(input_tensor)
            original_output = output
            output = torch.sigmoid(output)
            output = output * self.scaling_factor
            output = output + original_output
            return output

        if self.cudnn_flags and isinstance(self.cudnn_flags, dict):
            with torch.backends.cudnn.flags(**self.cudnn_flags):
                return _forward_impl(x)
        else:
            return _forward_impl(x)

batch_size = 128
input_size = 1024
hidden_size = 512
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]