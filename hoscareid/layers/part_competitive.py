import torch
import torch.nn.functional as F
from torch import nn


class PartCompetitiveBranch(nn.Module):
    """Build complementary local descriptors from HOSCA features."""

    def __init__(
        self,
        layer3_channels=1024,
        fused_channels=2048,
        num_parts=4,
        part_dim=256,
        num_classes=0,
        mask_temperature=1.0,
        vertical_prior_strength=1.0,
        vertical_prior_sigma=0.18,
    ):
        super().__init__()
        if num_parts < 2:
            raise ValueError("num_parts must be at least 2")
        if part_dim <= 0:
            raise ValueError("part_dim must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.num_parts = int(num_parts)
        self.part_dim = int(part_dim)
        self.mask_temperature = float(mask_temperature)
        self.vertical_prior_strength = float(vertical_prior_strength)
        self.vertical_prior_sigma = float(vertical_prior_sigma)

        self.mask_predictor = nn.Conv2d(
            fused_channels, self.num_parts, kernel_size=1, bias=True
        )
        self.layer3_proj = nn.Sequential(
            nn.Conv2d(layer3_channels, self.part_dim, 1, bias=False),
            nn.BatchNorm2d(self.part_dim),
            nn.ReLU(inplace=True),
        )
        self.fused_proj = nn.Sequential(
            nn.Conv2d(fused_channels, self.part_dim, 1, bias=False),
            nn.BatchNorm2d(self.part_dim),
            nn.ReLU(inplace=True),
        )

        gate_hidden = max(self.part_dim // 2, 32)
        self.reliability_gate = nn.Sequential(
            nn.Linear(self.part_dim * 4, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
        )

        descriptor_dim = self.num_parts * self.part_dim
        self.bottleneck = nn.BatchNorm1d(descriptor_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(descriptor_dim, num_classes, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.mask_predictor.weight)
        nn.init.zeros_(self.mask_predictor.bias)
        for module in (self.layer3_proj, self.fused_proj):
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for layer in self.reliability_gate.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.zeros_(self.reliability_gate[-1].bias)
        nn.init.ones_(self.bottleneck.weight)
        nn.init.zeros_(self.bottleneck.bias)
        nn.init.normal_(self.classifier.weight, std=0.01)

    def _vertical_prior(self, height, width, device, dtype):
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        centers = (
            torch.arange(self.num_parts, device=device, dtype=dtype) + 0.5
        ) / self.num_parts
        sigma = max(self.vertical_prior_sigma, 1e-3)
        prior = -((y.unsqueeze(0) - centers.unsqueeze(1)) / sigma).square()
        return prior.unsqueeze(-1).expand(self.num_parts, height, width)

    @staticmethod
    def _masked_pool(features, masks):
        weights = masks / masks.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return torch.einsum("bkhw,bdhw->bkd", weights, features)

    def forward(self, layer3_features, fused_features):
        if layer3_features.shape[-2:] != fused_features.shape[-2:]:
            layer3_features = F.adaptive_avg_pool2d(
                layer3_features, fused_features.shape[-2:]
            )

        mask_logits = self.mask_predictor(fused_features)
        prior = self._vertical_prior(
            mask_logits.shape[2],
            mask_logits.shape[3],
            mask_logits.device,
            mask_logits.dtype,
        )
        mask_logits = mask_logits + self.vertical_prior_strength * prior.unsqueeze(0)
        temperature = max(self.mask_temperature, 1e-3)
        masks = F.softmax(mask_logits / temperature, dim=1)

        layer3_projected = self.layer3_proj(layer3_features)
        fused_projected = self.fused_proj(fused_features)
        layer3_parts = self._masked_pool(layer3_projected, masks)
        fused_parts = self._masked_pool(fused_projected, masks)

        gate_input = torch.cat(
            [
                layer3_parts,
                fused_parts,
                torch.abs(layer3_parts - fused_parts),
                layer3_parts * fused_parts,
            ],
            dim=2,
        )
        reliability = torch.sigmoid(self.reliability_gate(gate_input))
        part_tokens = reliability * layer3_parts + (1.0 - reliability) * fused_parts

        raw_features = part_tokens.flatten(1)
        neck_features = self.bottleneck(raw_features)
        part_areas = masks.mean(dim=(2, 3))
        target_area = 1.0 / self.num_parts
        balance_loss = (part_areas - target_area).square().mean()

        outputs = {
            "features": neck_features,
            "raw_features": raw_features,
            "masks": masks,
            "reliability": reliability,
            "balance_loss": balance_loss,
        }
        if self.training:
            outputs["cls_outputs"] = self.classifier(neck_features)
        return outputs
