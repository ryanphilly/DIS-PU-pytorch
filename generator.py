import torch 
import torch.nn as nn
import . from layers

class FeatureExtractor(nn.Module):
  def __init__(self, in_channels, growth_rate, ):
    self.mlp1 = nn.Linear(in_channels, 24, bias=True)