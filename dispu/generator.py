import torch 
import torch.nn as nn

from math import log

from .layers import FeatureExtractor, DuplicateUp

class DISPUGenerator(torch.nn.Module):
    """Disentangled refinement upsampler"""
    def __init__(self, point_channels=3,
                       points_per_patch=234,
                       up_ratio=16,
                       step_ratio=2,
                       knn=16,
                       growth_rate=24,
                       dense_n=3,
                       fine_extractor=False,
                       refine=False,
                        **kwargs):
        self.up_ratio = up_ratio
        self.step_ratio = step_ratio
        self.knn = knn
        self.growth_rate = growth_rate
        self.points_per_patch = points_per_patch
        self.fine_extractor = fine_extractor
        self.refine = refine
        self.num_out_points = int(points_per_patch*up_ratio)

        self.feature_extractor = FeatureExtractor(
            point_channels=point_channels,
            dense_n=dense_n,
            growth_rate=growth_rate,
            knn=knn,
            step_ratio=step_ratio, **kwargs)

        self.duplicate_ups = nn.ModuleDict()
        for l in range(1, log(self.up_ratio, self.step_ratio)):
            self.duplicate_ups[str(l)] = DuplicateUp(, **kwargs)

        self.cordinate_regressor = CordinateRegressor(, **kwargs)

    def forward(self, xyz):
        # extract features
        coarse_feat = self.feature_extractor(xyz)
        patch_num = self.points_per_patch
        #loop log(up_ratio, step_size) and upsample (rQ)
        for l in range(1, log(self.up_ratio, self.step_ratio)):
            coarse_feat = self.duplicate_ups[str(l)](coarse_feat)
            patch_num = int(self.step_ratio*patch_num)

        # cordinate regression (delta Q)
        coarse = self.cordinate_regressor(coarse_feat)
        # refinement net
        pass
