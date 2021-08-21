from math import log

import torch 
import torch.nn as nn

from layers import (
    FeatureExtractor,
    DuplicateUp,
    CoordinateRegressor,
    PointShuffle)

class RefinedGenerator(torch.nn.Module):
    """Disentangled Point Cloud Up Sampler and Refiner"""
    def __init__(self, point_channels=3,
                       points_per_patch=324,
                       up_ratio=16,
                       step_ratio=2,
                       knn=16,
                       growth_rate=24,
                       dense_n=3,
                       fine_extractor=True,
                       refine=True,
                       offset=True,
                        **kwargs):
        super(RefinedGenerator, self).__init__()
        self.up_ratio = up_ratio
        self.step_ratio = step_ratio
        self.knn = knn
        self.growth_rate = growth_rate
        self.points_per_patch = points_per_patch
        self.fine_extractor = fine_extractor
        self.refine = refine
        self.offset = offset
        self.num_out_points = int(points_per_patch*up_ratio)

        self.coarse_feature_extractor = FeatureExtractor(
            point_channels=point_channels,
            dense_n=dense_n,
            growth_rate=growth_rate,
            knn=knn,
            step_ratio=step_ratio, **kwargs)

        self.duplicate_ups = nn.ModuleDict()
        for l in range(int(log(self.up_ratio, self.step_ratio))):
            if l != 0:
                self.duplicate_ups[str(l)] = DuplicateUp(input_channels=128, step_ratio=step_ratio)
            else:
                self.duplicate_ups[str(l)] = DuplicateUp(input_channels=480, step_ratio=step_ratio)

        self.coarse_coordinate_regressor = CoordinateRegressor(128, point_channels, **kwargs)

        if fine_extractor:
            self.fine_feature_extractor = FeatureExtractor(
                point_channels=point_channels,
                dense_n=dense_n,
                growth_rate=growth_rate,
                knn=knn,
                step_ratio=step_ratio, **kwargs)

        if refine:
            self.point_shuffle = PointShuffle(480+128, point_channels=point_channels, mlp_channels=[64,128])
            self.fine_coordinate_regressor = CoordinateRegressor(in_channels=128)

    def _generate_dense_cloud(self, points):
        """Dense Generator"""
        coarse_feat = self.coarse_feature_extractor(points)
        # feature expansion (rN)
        for l in range(int(log(self.up_ratio, self.step_ratio))):
            coarse_feat = self.duplicate_ups[str(l)](coarse_feat)

        coarse = self.coarse_coordinate_regressor(coarse_feat)
        return coarse, coarse_feat

    def _refine(self, coarse, coarse_feat):
        """Spatial Refiner"""
        if self.fine_extractor:
            fine_feat = self.fine_feature_extractor(coarse)
            fine_feat = torch.cat([fine_feat, coarse_feat], dim=1)
        else:
            fine_feat = coarse_feat

        if self.refine:
            new_coarse, fine_feat = self.point_shuffle(coarse, fine_feat, coarse)
            fine = self.fine_coordinate_regressor(fine_feat)
            if self.offset: fine = fine + new_coarse
        else:
            fine = coarse

        return fine, fine_feat

    def forward(self, points):
        coarse, coarse_feat = self._generate_dense_cloud(points)
        fine, _ = self._refine(coarse, coarse_feat)
        return coarse, fine


if __name__ == "__main__":
    sim_data = torch.rand((2, 3,  512))
    model = RefinedGenerator(point_channels=3, knn=16)
    print(model(sim_data)[1].shape)