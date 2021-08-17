from math import sqrt
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
    def __init__(self, input_channels=480, up_ratio=4):
        super(DuplicateUp,  self).__init__()
        self.up_ratio = up_ratio
        self.grid = self.make_grid(up_ratio)
        self.mlp_down1 = Conv2d(input_channels+2, 256, 1, activation='relu')
        self.mlp_down2 = Conv2d(256,  128, 1, activation='relu')

    def make_grid(self, up_ratio):
        grid_size = int(sqrt(up_ratio)) + 1
        x = torch.linspace(-0.2, 0.2, grid_size, dtype=torch.float32)
        x, y = torch.meshgrid(x, x)
        grid = torch.stack([x, y], dim=0).view([2, grid_size * grid_size])
        return grid.unsqueeze(0) # 1x2xgrid_size*grid_size

    def forward(self, x):
        B, _, N = x.size()
        _, _, ratio = self.grid.size()
        # 1x2xN*r
        code = self.grid.repeat(x.size(0), 1, N)
        code = code.to(device=x.device)
        # BxCxN -> BxCxNxr
        x = x.unsqueeze(-1).expand(-1, -1, -1, ratio)
        # BxCx(N*r)
        x = torch.reshape(x, [B, x.size(1), N * ratio]).contiguous()
        # Bx(C+2)xNr
        x = torch.cat([x, code], dim=1)

        x = x.unsqueeze(-1)
        # Bx128xNr
        x = self.mlp_down2(self.mlp_down1(x))
        x = torch.reshape(x, [B, x.size(1), N * ratio]).contiguous()
        return x

class CordinateRegressor(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

if __name__ == "__main__":
    sim_data = torch.rand((64, 3, 1024)).to('cuda:0')
    x = FeatureExtractor().to('cuda:0')
    y = DuplicateUp(edge=False).to('cuda:0')
    print(y(x(sim_data)).shape)