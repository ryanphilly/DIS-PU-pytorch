from math import sqrt

import torch 
import torch.nn as nn

from primitive_layers import DenseEdgeConv, Conv2d, Conv1d
from operations import group_knn, gather_nd

class FeatureExtractor(nn.Module):
    """3PU Feature Extractor"""
    def __init__(self, point_channels=3, dense_n=3, growth_rate=24, knn=16, step_ratio=2, **kwargs):
        super(FeatureExtractor, self).__init__()
        self.dense_n = dense_n
        self.step_ratio = step_ratio
        comp = growth_rate*2
        # TODO: make in_channels calculate channels dynamicaly based on growth rate
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
        B x point_channels x N -> BxCxN
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
    def __init__(self, input_channels=480, step_ratio=2):
        super(DuplicateUp,  self).__init__()
        self.step = step_ratio
        self.grid = self.make_grid(step_ratio)
        input_channels = input_channels+2 if step_ratio >= 4 else input_channels+1
        self.mlp_down1 = Conv2d(input_channels, 256, 1, activation='relu')
        self.mlp_down2 = Conv2d(256,  128, 1, activation='relu')

    def make_grid(self, step_ratio):
        grid_size = int(sqrt(step_ratio)) + 1
        x = torch.linspace(-0.2, 0.2, grid_size, dtype=torch.float32)
        if step_ratio >= 4:
            x, y = torch.meshgrid(x, x)
            grid = torch.stack([x, y], dim=0).view([2, grid_size * grid_size])
        else:
            grid = x.view(1, grid_size)
        return grid.unsqueeze(0) # 1x1xgrid_size or 1x2xgrid_size^2

    def forward(self, x):
        # B x C x N -> B x 128 x rN
        batch_size, _, num_points = x.size()
        _, _, up_ratio = self.grid.size()
        # 1 x 2 x N*r
        grid = self.grid.repeat(x.size(0), 1, num_points)
        grid = grid.to(device=x.device)
        # B x C x N x r
        x = x.unsqueeze(-1).expand(-1, -1, -1, up_ratio)
        # B x C x (N*r)
        x = torch.reshape(x, [batch_size, x.size(1), num_points * up_ratio]).contiguous()
        # B x (C+2) x Nr
        x = torch.cat([x, grid], dim=1)
        x = x.unsqueeze(-1)
        # B x 128 x Nr
        x = self.mlp_down2(self.mlp_down1(x))
        x = x.view((batch_size, x.size(1), num_points * up_ratio)).contiguous()
        return x

class CoordinateRegressor(nn.Module):
    def __init__(self, in_channels=128, out_channels=3):
        super(CoordinateRegressor, self).__init__()
        self.lin1 = Conv1d(in_channels, 256, 1, activation='relu')
        self.lin2 = Conv1d(256, 64, 1, activation='relu')
        self.lin3 = Conv1d(64, out_channels, 1)

    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(x)))

class PointShuffle(nn.Module):
    def __init__(self, in_channels, point_channels=3, mlp_channels=[128,128,256], knn=16, use_points=True, refine_points=True, non_local=True, local=True):
        super(PointShuffle, self).__init__()
        self.knn = knn
        self.in_channels = in_channels+point_channels if not use_points else in_channels+point_channels+point_channels
        self.out_channels = mlp_channels[-1]
        self.use_points = use_points
        self.refine_points = refine_points
        self.non_local = non_local
        self.local = local

        self.spatial_skip_mlp = Conv1d(self.in_channels, self.out_channels, 1, activation='relu')
        self.mlps = nn.ModuleList()
        self.mlps.append(Conv2d(self.in_channels, mlp_channels[0], 1, activation='relu'))
        for l in range(1, len(mlp_channels)):
            self.mlps.append(Conv2d(mlp_channels[l-1], mlp_channels[l], 1, activation='relu'))

        self.hidden_net = Conv2d(point_channels, knn, 1, activation='relu')
        self.output_mlp1 = Conv2d(knn, self.out_channels, kernel_size=[1, self.out_channels], activation='relu')
        self.output_mlp2 = Conv1d(self.out_channels, self.out_channels, 1, activation='relu')   

        
    def _grouping(self, points, point_features, query_points):
        batch_size, _, num_points = query_points.size()
        # B x M x knn
        _, idx, _ = group_knn(self.knn, query_points, points)
        # B x 1 x M x knn
        batch_indices = torch.arange(0, batch_size, 1).view(-1, 1, 1, 1)
        batch_indices = torch.tile(batch_indices, (1, 1, num_points, self.knn))
        # B x 2 x M x knn
        idx =  torch.cat([batch_indices, idx.unsqueeze(1)], dim=1)
        idx = idx.view((batch_size, 2, num_points, self.knn))
        # B x point_channels x N x knn
        grouped_points = gather_nd(points, idx)
        # B x C x N x knn
        grouped_features = gather_nd(point_features, idx) 
        if self.use_points:
            grouped_features = torch.cat([grouped_points, grouped_features], dim=1)

        return grouped_points, grouped_features, idx

    def forward(self, points, point_features, query_points):
        new_points = points
        grouped_points, grouped_features, idx = self._grouping(points, point_features, query_points)
  
        grouped_points -= torch.tile(new_points.unsqueeze(3), (1, 1, 1, self.knn))
        grouped_features = torch.cat([grouped_features, grouped_points], dim=1)
   
        if self.refine_points:
            pass

        if self.non_local:
            pass
        
        spatial_skip_connecttion, _ = torch.max(grouped_features, dim=3, keepdim=False)
        spatial_skip_connecttion = self.spatial_skip_mlp(spatial_skip_connecttion)
        for mlp in self.mlps:
            grouped_features = mlp(grouped_features)

        '''
        weight = self.hidden_net(grouped_points)
        print(weight.shape)
        grouped_features = torch.transpose(grouped_features, 3, 1)
        print(grouped_features.shape)
        grouped_features = grouped_features @ weight
        print(grouped_features.shape)
        '''
        grouped_features = grouped_features.transpose(1, 3)
        grouped_features = self.output_mlp1(grouped_features)
        grouped_features = grouped_features.squeeze(3)
        grouped_features = grouped_features + spatial_skip_connecttion
        grouped_features = self.output_mlp2(grouped_features)

        return new_points, grouped_features


if __name__ == "__main__":
    sim_points = torch.rand((64, 3, 1024))
    sim_feats = torch.rand((64, 24, 1024))
    model = PointShuffle(24, mlp_channels=[64, 128])
    print(model(sim_points, sim_feats, sim_points)[1].shape)

  