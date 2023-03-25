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

class DepthToSemantic(nn.Module):
    def __init__(self, class_num) -> None:
        super(DepthToSemantic, self).__init__()
        self.depth_feature_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) 

        self.com_rep_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) 

        self.depth_feature_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        ) 

        self.com_rep_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.pixel_mul_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.sem_feature_conv = nn.Conv2d(in_channels=160, out_channels=class_num, kernel_size=3, padding=1)

        self.upconv8 = UpConv2D(in_channels=class_num, out_channels=class_num, scale=8)

    def forward(self, semantic_feature, common_rep, depth_feature):
        depth_feature = self.depth_feature_conv1(depth_feature)
        depth_feature = self.depth_feature_conv2(depth_feature)

        common_rep = self.com_rep_conv1(common_rep)
        common_rep = self.com_rep_conv2(common_rep)

        semantic = self.sem_feature_conv(
            torch.cat(
                tensors=(semantic_feature,self.pixel_mul_conv(common_rep*depth_feature)),
                dim=1
            )
        )
        semantic = self.upconv8(semantic)
        return semantic

