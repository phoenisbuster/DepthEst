import torch
import torch.nn as nn
from torch.nn import functional as F

class UpConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(UpConv2D, self).__init__()
        self.scale = scale

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        if self.scale != 1:
            out = F.interpolate(input=out, scale_factor=self.scale, mode='bilinear')
        
        residual = self.residual(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.relu(out + residual)

        return out

class SematicToDepth(nn.Module):
    def __init__(self) -> None:
        super(SematicToDepth, self).__init__()
        self.sem_feature_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        ) 

        self.com_rep_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        ) 

        self.sem_feature_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        ) 

        self.com_rep_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.depth_feature_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.depth_feature_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=1, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.upconv8 = UpConv2D(in_channels=1, out_channels=1, scale=8)

    def forward(self, semantic_feature, common_rep, depth_feature):
        semantic_feature = self.sem_feature_conv1(semantic_feature)
        semantic_feature = self.sem_feature_conv2(semantic_feature)

        common_rep = self.com_rep_conv1(common_rep)
        common_rep = self.com_rep_conv2(common_rep)

        depth = self.depth_feature_conv1(depth_feature)
        depth = self.depth_feature_conv2(
            torch.cat(
                tensors=(torch.sqrt(semantic_feature*common_rep), depth),
                dim=1
            )
        )
        depth = self.upconv8(depth)
        return depth

