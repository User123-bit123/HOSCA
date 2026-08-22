import torch
from torch import nn
import torch.nn.functional as F
from .coord_att import CoordAtt


class LightweightOSBlock_CA(nn.Module):
    """Stage-3 lightweight OS-CA block used by HOSCA."""

    def __init__(self, in_channels, out_channels, reduction=16):
        super(LightweightOSBlock_CA, self).__init__()
        mid_channels = out_channels // 4
        
        # Branch 1: 1x1 conv
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv33 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: 5x5 conv (dilated)
        self.conv55 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 4: AvgPool + 1x1 
        self.avgpool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
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
        x4 = self.avgpool(x)
        
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.ca(out)
        
        return F.relu(out + identity)


class HierarchicalOSCA(nn.Module):
    """OS-CA refinement and gated residual fusion for Stage-3/Stage-4 features."""

    def __init__(self, layer3_channels=1024, layer4_channels=2048, reduction=16):
        super(HierarchicalOSCA, self).__init__()
        self.osca_layer3 = LightweightOSBlock_CA(layer3_channels, layer3_channels, reduction)

        from .os_ca_block import OSBlock_CA
        self.osca_layer4 = OSBlock_CA(layer4_channels, layer4_channels, reduction)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(layer3_channels + layer4_channels, layer4_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(layer4_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(layer3_channels + layer4_channels, 2, 1),
            # nn.BatchNorm2d(2),
            nn.Sigmoid()
        )

    def forward(self, features_dict, return_aux=False):
        """
        Args:
            features_dict: dict with 'layer3' and 'layer4' feature maps.
            return_aux: if True, also return the aligned Stage-3 feature and
                the fused feature for the PCL branch.
        Returns:
            fused_features, or a dictionary containing the fused feature and
            the intermediate features when return_aux is True.
        """
        feat_l3 = features_dict['layer3']
        feat_l4_raw = features_dict['layer4']

        feat_l3 = self.osca_layer3(feat_l3)
        feat_l4 = self.osca_layer4(feat_l4_raw)

        feat_l3_down = F.adaptive_avg_pool2d(feat_l3, feat_l4.shape[2:])
        combined = torch.cat([feat_l3_down, feat_l4], dim=1)

        weights = self.fusion_gate(combined)
        w_l3, w_l4 = weights[:, 0:1], weights[:, 1:2]

        feat_l3_weighted = feat_l3_down * w_l3
        feat_l4_weighted = feat_l4 * w_l4
        fused = self.fusion_conv(torch.cat([feat_l3_weighted, feat_l4_weighted], dim=1))

        fused = F.relu(fused + feat_l4_raw)
        if not return_aux:
            return fused

        return {
            "fused": fused,
            "layer3": feat_l3_down,
            "layer4": feat_l4,
            "layer4_raw": feat_l4_raw,
            "gate": weights,
        }
