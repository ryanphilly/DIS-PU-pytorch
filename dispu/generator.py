import torch 
import torch.nn as nn


class DISPUGenerator(torch.nn.Module):
    """Disentangled refinement upsampler"""
    def __init__(self, up_ratio=16, step_ratio=2, knn=16, growth_rate=24, dense_n=3, **kwargs):
        pass

    def forward(self, xyz):
        # extract features

        #loop log(up_ratio, step_size) and upsample (Q`)

        # cordinate regression (delta Q)

        # refinement net
        pass
