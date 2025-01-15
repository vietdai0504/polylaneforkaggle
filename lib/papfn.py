import torch
import torch.nn as nn
import torch.nn.functional as F

class PathAggregationFeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(PathAggregationFeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x):
        results = []
        last_inner = self.inner_blocks[-1](x[str(len(self.inner_blocks) - 1)])
        results.append(self.layer_blocks[-1](last_inner))
        for idx in range(len(self.inner_blocks) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[str(idx)])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        return {str(i): results[i] for i in range(len(results))}
