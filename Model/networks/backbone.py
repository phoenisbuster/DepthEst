import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(SepConv2D, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size, 
            groups=in_channels,
            bias=bias,
            padding=1
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=1,
            bias=bias
        )
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    

class SepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsameple=False, residual=False) -> None:
        super(SepBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsameple = downsameple
        self.residual = residual

        self.sepConv1 = SepConv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.sepConv2 = SepConv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.sepConv3 = SepConv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsamepleConv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        out = self.relu1(self.bn1(self.sepConv1(x)))
        out = self.relu2(self.bn2(self.sepConv2(out)))
        out = self.bn3(self.sepConv3(out))
        if self.downsameple:
            out = self.downsamepleConv(self.relu3(out))
            out = self.bn4(out)
        if self.residual:
            out = self.relu3(out + x)
        return out


class ImagePooling(nn.Module):
    def __init__(self) -> None:
        super(ImagePooling, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        _, _, H, W = x.shape

        first_branch = self.mean(x)
        first_branch = F.normalize(input=first_branch, p=2, dim=1)
        first_branch = torch.tile(first_branch, (1,1,H,W))

        second_branch = F.normalize(input=x, p=2, dim=(2,3))

        out = torch.cat(tensors=(second_branch, first_branch), dim=1)
        return out
    

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ASPP, self).__init__()

        self.img_pooling = ImagePooling()
        self.atrousConv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels)
        )

        self.atrousConv6 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(num_features=out_channels)
        )

        self.atrousConv12 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(num_features=out_channels)
        )

        self.atrousConv18 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=18, dilation=18),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2+out_channels*4, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        img_features = self.img_pooling(x)
        atrous1 = self.atrousConv1(x)
        atrous6 = self.atrousConv6(x)
        atrous12 = self.atrousConv12(x)
        atrous18 = self.atrousConv18(x)
        out = torch.cat(tensors=(img_features, atrous1, atrous6, atrous12, atrous18), dim=1)
        out = self.conv(out)
        return out


class Backbone(nn.Module):
    def __init__(self) -> None:
        super(Backbone, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # block 1
        self.residualConv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128)
        )
        self.sepConvBlock1 = SepBlock(in_channels=64, out_channels=128, kernel_size=3, downsameple=True)

        # block 2
        self.residualConv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=256)
        )
        self.sepConvBlock2 = SepBlock(in_channels=128, out_channels=256, kernel_size=3, downsameple=True)

        # block 3
        self.residualConv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=728, kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=728)
        )
        self.sepConvBlock3 = SepBlock(in_channels=256, out_channels=728, kernel_size=3, downsameple=True)

        # block 4
        self.sepConvBlock4 = nn.Sequential()
        for _ in range(16):
            self.sepConvBlock4.append(SepBlock(in_channels=728, out_channels=728, kernel_size=3, residual=True))

        # block 5
        self.residualConv5 = nn.Sequential(
            nn.Conv2d(in_channels=728, out_channels=1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=1024)
        )
        self.sepConvBlock5 = SepBlock(in_channels=728, out_channels=1024, kernel_size=3, downsameple=True)

        self.sepConv1 = nn.Sequential(
            SepConv2D(in_channels=1024, out_channels=1536, kernel_size=3),
            nn.BatchNorm2d(num_features=1536),
            nn.ReLU(inplace=True)
        )

        self.sepConv2 = nn.Sequential(
            SepConv2D(in_channels=1536, out_channels=1536, kernel_size=3),
            nn.BatchNorm2d(num_features=1536),
            nn.ReLU(inplace=True)
        )

        self.sepConv3 = nn.Sequential(
            SepConv2D(in_channels=1536, out_channels=2048, kernel_size=3),
            nn.BatchNorm2d(num_features=2048),
            nn.ReLU(inplace=True)
        )

        # ASPP
        self.aspp = ASPP(in_channels=2048, out_channels=512)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        # block 1
        residual = self.residualConv1(out)
        out = self.sepConvBlock1(out)
        out = nn.ReLU(inplace=True)(out + residual)
        residual = nn.ReLU(inplace=True)(residual)
        
        # block 2
        residual = self.residualConv2(residual)
        out = self.sepConvBlock2(out)
        out = nn.ReLU(inplace=True)(out + residual)
        residual = nn.ReLU(inplace=True)(residual)

        refined_fp = out

        # block 3
        residual = self.residualConv3(residual)
        out = self.sepConvBlock3(out)
        out = nn.ReLU(inplace=True)(out + residual)

        # block 4
        out = self.sepConvBlock4(out)
        residual = out
        
        # block 5
        residual = self.residualConv5(residual)
        out = self.sepConvBlock5(out)
        out = nn.ReLU(inplace=True)(out + residual)
        
        # block 6
        out = self.sepConv1(out)
        out = self.sepConv2(out)
        out = self.sepConv3(out)

        # ASPP
        out = self.aspp(out)

        return out, refined_fp