import torch
from torch import nn
import torch.nn.functional as F
from .coord_att import CoordAtt

class OSBlock_CA(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(OSBlock_CA, self).__init__()
        mid_channels = out_channels // 4
        
        # Branch 1: 1x1 conv
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 3x3 conv (using two 3x3 to simulate larger receptive field)
        self.conv33 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 5x5 conv (using 3x3 atrous conv or multiple 3x3)
        self.conv55 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 4: MaxPool + 1x1
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion and Attention
        self.ca = CoordAtt(out_channels, out_channels, reduction=reduction)
        
        # Downsample if needed
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
            
        x1 = self.conv11(x)
        x2 = self.conv33(x)
        x3 = self.conv55(x)
        x4 = self.maxpool(x)
        
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.ca(out)
        
        return F.relu(out + identity)
