import torch 
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, global_max_pool


def MLP(channels, **kwargs):
    return nn.Sequential(*[nn.Sequential(
      nn.Linear(channels[i - 1], channels[i], **kwargs),
      nn.ReLU(),
      nn.BatchNorm1d(channels[i])) for i in range(1, len(channels))
    ])

class DenseDynamicEdgeConv(nn.Module):
  """DGCNN -> linear -> linear -> etc (only the first"""
  def __init__(self, in_channels, num_layers=3, growth_rate=64, knn=16, **kwargs):
    self.k, self.in_channels = knn, in_channels
    self.layers = nn.ModuleList()
    # only first layer dynamicaly costructs the graph from knn search
    self.layers.append(DynamicEdgeConv(MLP([in_channels*2, growth_rate], **kwargs)))
    for l in range(1, num_layers):
      in_channels += growth_rate
      self.layers.append(MLP([in_channels, growth_rate], **kwargs))

  def forward(self, x, batch):
    for l, layer in enumerate(self.layers):
      if l == 0:
        x = torch.cat([layer(x, batch), x], dim=1)
      else:
        z = torch.cat([layer(x), x], dim=1)

    return global_max_pool(x, batch)



    