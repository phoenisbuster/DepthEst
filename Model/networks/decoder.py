import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    nn.init.xavier_uniform_(m.weight)

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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.upsameple4 = UpConv2D(in_channels=512, out_channels=128, scale=4)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.semantic_feature_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.depth_feature_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

    def forward(self, backbone_out, refined_fp):
        common_rep = self.conv1(backbone_out)
        common_rep = self.upsameple4(common_rep)

        semantic_feature = self.conv2(refined_fp)
        semantic_feature = self.semantic_feature_conv(
            torch.cat(
                tensors=(semantic_feature,common_rep),
                dim=1
            )
        )

        depth_feature = self.conv3(refined_fp)
        depth_feature = self.depth_feature_conv(
            torch.cat(
                tensors=(common_rep, depth_feature),
                dim=1
            )
        )

        return semantic_feature, common_rep, depth_feature
