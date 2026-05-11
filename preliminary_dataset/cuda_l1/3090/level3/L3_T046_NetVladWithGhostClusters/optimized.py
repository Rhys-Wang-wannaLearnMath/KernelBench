import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size
        
        # Pre-compute batch norm parameters for maximum efficiency
        self.register_buffer('bn_weight', None)
        self.register_buffer('bn_bias', None)
        self.register_buffer('bn_mean', None)
        self.register_buffer('bn_var_sqrt_inv', None)
        
    def _update_bn_params(self):
        """Pre-compute batch norm parameters for efficient forward pass"""
        if (self.bn_weight is None or 
            self.bn_weight.device != self.clusters.device or
            not self.bn_weight.is_contiguous()):
            
            eps = self.batch_norm.eps
            self.bn_weight = self.batch_norm.weight.contiguous()
            self.bn_bias = self.batch_norm.bias.contiguous()
            self.bn_mean = self.batch_norm.running_mean.contiguous()
            self.bn_var_sqrt_inv = torch.rsqrt(self.batch_norm.running_var + eps).contiguous()

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (torch.Tensor): B x N x D

        Returns:
            (torch.Tensor): B x DK
        """
        batch_size, max_sample, _ = x.shape
        
        if x.device != self.clusters.device:
            msg = f"x.device {x.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)
        
        # Update batch norm parameters
        self._update_bn_params()
        
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Flatten input for matrix multiplication
        x_flat = x.view(-1, self.feature_size)  # BN x D
        
        # Compute assignment using optimized matrix multiplication
        assignment = torch.mm(x_flat, self.clusters)  # BN x (K+G)
        
        # Apply batch normalization manually for efficiency
        assignment = torch.addcmul(
            self.bn_bias,
            assignment - self.bn_mean,
            self.bn_weight * self.bn_var_sqrt_inv
        )
        
        # Apply softmax and slice to remove ghost clusters
        assignment = F.softmax(assignment, dim=1)[:, :self.cluster_size]
        
        # Reshape assignment back to batch format
        assignment = assignment.view(batch_size, max_sample, self.cluster_size)
        
        # Compute sum of assignments for each cluster
        a_sum = torch.sum(assignment, dim=1, keepdim=True)  # B x 1 x K
        
        # Compute weighted cluster centers
        a = a_sum * self.clusters2  # B x D x K
        
        # Optimize VLAD computation by transposing x once
        x_t = x.transpose(1, 2)  # B x D x N
        
        # Use batch matrix multiplication for VLAD computation
        vlad = torch.bmm(x_t, assignment)  # B x D x K
        
        # Subtract cluster centers in-place
        vlad.sub_(a)  # B x D x K
        
        # L2 intra-normalization
        vlad = F.normalize(vlad, p=2, dim=1)
        
        # Flatten and apply final L2 normalization
        vlad = vlad.reshape(batch_size, -1)  # B x DK
        vlad = F.normalize(vlad, p=2, dim=1)
        
        return vlad

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 32
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 16

def get_inputs():
    return [torch.randn(batch_size, num_features, feature_size)]

def get_init_inputs():
    return [num_clusters, feature_size, ghost_clusters]