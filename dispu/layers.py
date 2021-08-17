import torch 
import torch.nn as nn
from utils.base_layers import DenseEdgeConv, Conv2d, Conv1d

class FeatureExtractor(nn.Module):
    """3PU Feature Extractor"""
    def __init__(self, point_channels=3, dense_n=3, growth_rate=24, knn=16, step_ratio=2, **kwargs):
        super(FeatureExtractor, self).__init__()
        self.dense_n = dense_n
        self.step_ratio = step_ratio
        comp = growth_rate*2
        # TODO: make in_channels calculate channels dynamicaly based on growth rate
        # currently only works for growth_rate = 24
        in_channels = point_channels
        self.layer0 = Conv2d(in_channels, 24, [1, 1], activation=None)
        self.layer1 = DenseEdgeConv(24, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 120
        self.layer2_prep = Conv1d(in_channels, comp, 1, activation="relu")
        self.layer2 = DenseEdgeConv(comp, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 240
        self.layer3_prep = Conv1d(in_channels, comp, 1, activation="relu")
        self.layer3 = DenseEdgeConv(comp, growth_rate=growth_rate, n=dense_n, k=knn)
        in_channels = 360 
        self.layer4_prep = Conv1d(in_channels, comp, 1, activation="relu")
        self.layer4 = DenseEdgeConv(comp, growth_rate=growth_rate, n=dense_n, k=knn)

    def forward(self, xyz):
        """
        :param
            xyz  B x point_channels x N input xyz, normalized
        :returns
            xyz_featues BxCxN
        """
        x = self.layer0(xyz.unsqueeze(dim=-1)).squeeze(dim=-1)
        y, _ = self.layer1(x)
        x = torch.cat([y, x], dim=1)
        y, _ = self.layer2(self.layer2_prep(x))
        x = torch.cat([y, x], dim=1)
        y, _ = self.layer3(self.layer3_prep(x))
        x = torch.cat([y, x], dim=1)
        y, _ = self.layer4(self.layer4_prep(x))
        x = torch.cat([y, x], dim=1)
        return x

class DuplicateUp(nn.Module):
    def __init__(self, up_ratio=4):
        pass
    
    def make_grid(self):
        pass

    def forward(self, x):
        pass

if __name__ == "__main__":
    sim_data = torch.rand((64, 3, 1024)).to('cuda:0')
    x= FeatureExtractor().to('cuda:0')
    print(x(sim_data).shape)