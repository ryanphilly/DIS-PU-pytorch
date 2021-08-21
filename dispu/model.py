import torch
from generator import RefinedGenerator

class Model(torch.nn.Module):
  def __init__(self):
    self.generator = RefinedGenerator()