"""Author: Yifan Wang  yifan.wang@inf.ethz.ch"""
import torch
import torch.nn as nn
from .operations import group_knn

class DenseEdgeConv(nn.Module):
    """docstring for EdgeConv"""

    def __init__(self, in_channels, growth_rate, n, k, **kwargs):
        super(DenseEdgeConv, self).__init__()
        self.growth_rate = growth_rate
        self.n = n
        self.k = k
        self.mlps = torch.nn.ModuleList()
        self.mlps.append(torch.nn.Conv2d(
            2 * in_channels, growth_rate, 1, bias=True))
        for i in range(1, n):
            in_channels += growth_rate
            self.mlps.append(torch.nn.Conv2d(
                in_channels, growth_rate, 1, bias=True))

    def get_local_graph(self, x, k, idx=None):
        """Construct edge feature [x, NN_i - x] for each point x
        :param
            x: (B, C, N)
            k: int
            idx: (B, N, k)
        :return
            edge features: (B, C, N, k)
        """
        if idx is None:
            # BCN(K+1), BN(K+1)
            knn_point, idx, _ = group_knn(k + 1, x, x, unique=True)
            idx = idx[:, :, 1:]
            knn_point = knn_point[:, :, :, 1:]

        neighbor_center = torch.unsqueeze(x, dim=-1)
        neighbor_center = neighbor_center.expand_as(knn_point)

        edge_feature = torch.cat(
            [neighbor_center, knn_point - neighbor_center], dim=1)
        return edge_feature, idx

    def forward(self, x, idx=None):
        """
        args:
            x features (B,C,N)
        return:
            y features (B,C',N)
            idx fknn index (B,C,N,K)
        """
        # [B 2C N K]
        for i, mlp in enumerate(self.mlps):
            if i == 0:
                y, idx = self.get_local_graph(x, k=self.k, idx=idx)
                x = x.unsqueeze(-1).repeat(1, 1, 1, self.k)
                y = torch.cat([nn.functional.relu_(mlp(y)), x], dim=1)
            elif i == (self.n - 1):
                y = torch.cat([mlp(y), y], dim=1)
            else:
                y = torch.cat([nn.functional.relu_(mlp(y)), y], dim=1)

        y, _ = torch.max(y, dim=-1)
        return y, idx
      
class Conv2d(nn.Module):
    """2dconvolution with custom normalization and activation"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 activation=None, normalization=None, momentum=0.01):
        super(Conv2d, self).__init__()
        self.activation = activation
        self.normalization = normalization
        bias = not normalization and bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)

        if normalization is not None:
            if self.normalization == 'batch':
                self.norm = nn.BatchNorm2d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            elif self.normalization == 'instance':
                self.norm = nn.InstanceNorm2d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            else:
                raise ValueError(
                    "only \"batch/instance\" normalization permitted.")

        # activation
        if activation is not None:
            if self.activation == 'relu':
                self.act = nn.ReLU()
            elif self.activation == 'elu':
                self.act = nn.ELU(alpha=1.0)
            elif self.activation == 'lrelu':
                self.act = nn.LeakyReLU(0.1)
            else:
                raise ValueError("only \"relu/elu/lrelu\" allowed")

    def forward(self, x, epoch=None):
        x = self.conv(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x


class Conv1d(nn.Module):
    """1dconvolution with custom normalization and activation"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 activation=None, normalization=None, momentum=0.01):
        super(Conv1d, self).__init__()
        self.activation = activation
        self.normalization = normalization
        bias = not normalization and bias
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)

        if normalization is not None:
            if self.normalization == 'batch':
                self.norm = nn.BatchNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            elif self.normalization == 'instance':
                self.norm = nn.InstanceNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            else:
                raise ValueError(
                    "only \"batch/instance\" normalization permitted.")

        # activation
        if activation is not None:
            if self.activation == 'relu':
                self.act = nn.ReLU()
            elif self.activation == 'elu':
                self.act = nn.ELU(alpha=1.0)
            elif self.activation == 'lrelu':
                self.act = nn.LeakyReLU(0.1)
            else:
                raise ValueError("only \"relu/elu/lrelu\" allowed")

    def forward(self, x, epoch=None):
        x = self.conv(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x
