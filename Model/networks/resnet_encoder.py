import torch
import torch.nn as nn
from torch import Tensor

from typing import Any, Callable, List, Optional, Type, Union

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class Block(nn.Module):
    def __init__(self, in_channels, channels, stride=1, is_conv_block=False):
        super(Block, self).__init__()

        self.is_conv_block = is_conv_block

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=4*channels, kernel_size=1, stride=1)
        
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.bn2 = nn.BatchNorm2d(num_features=channels)
        self.bn3 = nn.BatchNorm2d(num_features=4*channels)
        
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if is_conv_block:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=4*channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(4*channels),
            )
            self.downsample.apply(init_weights)

        init_weights(self.conv1)
        init_weights(self.conv2)
        init_weights(self.conv3)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.is_conv_block:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Encoder(nn.Module):
    def __init__(self, H=160, W=608):
        super(ResNet_Encoder, self).__init__()

        self.H = H
        self.W = W

        self.inplanes = 64
        
        # stage 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage 2-5
        self.stage2 = self.make_stage(in_channels=64, channels=64, num_blocks=3, is_stage2=True)
        self.stage3 = self.make_stage(in_channels=256, channels=128, num_blocks=4)
        self.stage4 = self.make_stage(in_channels=512, channels=256, num_blocks=6)
        self.stage5 = self.make_stage(in_channels=1024, channels=512, num_blocks=3)

        init_weights(self.conv1)
        
    def make_stage(self, in_channels, channels, num_blocks, is_stage2=False):
        stage = nn.Sequential()
        if is_stage2:
            stage.append(Block(in_channels=in_channels, channels=channels, stride=1, is_conv_block=True))
        else:
            stage.append(Block(in_channels=in_channels, channels=channels, stride=2, is_conv_block=True))

        for _ in range(1, num_blocks):
            stage.append(Block(in_channels=4*channels, channels=channels))
        return stage

    def forward(self, x):
        self.features = []
        
        x = self.conv1(x)
        x = self.bn(x)
        self.features.append(self.relu(x)) # 80
        self.features.append(self.maxpool(self.features[-1])) # 40

        self.features.append(self.stage2(self.features[-1])) # 40
        self.features.append(self.stage3(self.features[-1])) # 20
        self.features.append(self.stage4(self.features[-1])) # 10
        self.features.append(self.stage5(self.features[-1])) # 5

        return self.features



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def resnet50(weight=None):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if weight:
        model.load_state_dict(torch.load(weight))
    return model

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Bottleneck,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Bottleneck,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        self.features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        self.features.append(self.relu(x)) # 80
        self.features.append(self.maxpool(self.features[-1])) # 40

        self.features.append(self.layer1(self.features[-1])) # 40
        self.features.append(self.layer2(self.features[-1])) # 20
        self.features.append(self.layer3(self.features[-1])) # 10
        self.features.append(self.layer4(self.features[-1])) # 5

        return self.features