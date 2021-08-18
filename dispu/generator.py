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
                       points_per_patch=234,
                       up_ratio=16,
                       step_ratio=2,
                       knn=16,
                       growth_rate=24,
                       dense_n=3,
                       fine_extractor=True,
                       refine=True,
                       is_off=True,
                        **kwargs):
        super(RefinedGenerator, self).__init__()
        self.up_ratio = up_ratio
        self.step_ratio = step_ratio
        self.knn = knn
        self.growth_rate = growth_rate
        self.points_per_patch = points_per_patch
        self.fine_extractor = fine_extractor
        self.refine = refine
        self.is_off = is_off
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
                self.duplicate_ups[str(l)] = DuplicateUp(input_channels=128, up_ratio=2)
            else:
                self.duplicate_ups[str(l)] = DuplicateUp(input_channels=480, up_ratio=2)

        self.coarse_coordinate_regressor = CoordinateRegressor(128, point_channels, **kwargs)

        if fine_extractor:
            self.fine_feature_extractor = FeatureExtractor(
                point_channels=point_channels,
                dense_n=dense_n,
                growth_rate=growth_rate,
                knn=knn,
                step_ratio=step_ratio, **kwargs)

        if refine:
            self.point_shuffle = PointShuffle()
            self.fine_coordinate_regressor = CoordinateRegressor()

    def _generate_upsampled_cloud(self, points):
        """Upsample network"""
        coarse_feat = self.coarse_feature_extractor(points)
        # upsample (rN)
        for l in range(int(log(self.up_ratio, self.step_ratio))):
            coarse_feat = self.duplicate_ups[str(l)](coarse_feat)

        coarse = self.coarse_coordinate_regressor(coarse_feat)
        return coarse, coarse_feat

    def _refine(self, coarse, coarse_feat):
        """Refinement network"""
        if self.fine_extractor:
            fine_feat = self.fine_feature_extractor(coarse)
            fine_feat = torch.cat([fine_feat, coarse_feat], dim=1)
        else:
            fine_feat = coarse_feat

        if self.refine:
            new_coarse, fine_feat = self.point_shuffle(coarse, fine_feat)
            fine = self.fine_coordinate_regressor(fine_feat)
            if self.is_off:
                fine = fine + new_coarse
        else:
            fine = coarse

        return fine, fine_feat

    def forward(self, points):
        coarse, coarse_feat = self._generate_upsampled_cloud(points)
        fine, _ = self._refine(coarse, coarse_feat)
        return coarse, fine


if __name__ == "__main__":
    sim_data = torch.rand((1, 3,  116))
    model = RefinedGenerator()
    print(model(sim_data).shape)