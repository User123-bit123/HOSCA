# encoding: utf-8
"""
Minimal Baseline meta-architecture for the HOSCA release pack.

This trimmed version keeps only the code path required by the final
`bagtricks_R50_IBN_HOSCA_GeM_v8_map_opt` experiment:

- hierarchical backbone feature extraction
- HierarchicalOSCA attention/fusion
- classification + triplet supervision
"""

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import (
    cross_entropy_loss,
    log_accuracy,
    triplet_loss,
)
from fastreid.layers.hierarchical_osca import HierarchicalOSCA
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Minimal Baseline architecture for the HOSCA release.

    The release pack only preserves the reproduction path used in the paper:
    backbone -> HierarchicalOSCA -> embedding head -> CE + Triplet losses.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone,
        heads,
        pixel_mean,
        pixel_std,
        loss_kwargs=None,
        layers_attn=None,
        use_hierarchical=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.attn = layers_attn
        self.use_hierarchical = use_hierarchical
        self.loss_kwargs = loss_kwargs or {}

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)

        layers_attn = None
        use_hierarchical = False
        if cfg.MODEL.BACKBONE.WITH_HOSCA:
            layers_attn = HierarchicalOSCA(layer3_channels=1024, layer4_channels=2048)
            use_hierarchical = True

        return {
            "backbone": backbone,
            "heads": heads,
            "layers_attn": layers_attn,
            "use_hierarchical": use_hierarchical,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "loss_kwargs": {
                "loss_names": cfg.MODEL.LOSSES.NAME,
                "ce": {
                    "eps": cfg.MODEL.LOSSES.CE.EPSILON,
                    "alpha": cfg.MODEL.LOSSES.CE.ALPHA,
                    "scale": cfg.MODEL.LOSSES.CE.SCALE,
                },
                "tri": {
                    "margin": cfg.MODEL.LOSSES.TRI.MARGIN,
                    "norm_feat": cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                    "hard_mining": cfg.MODEL.LOSSES.TRI.HARD_MINING,
                    "scale": cfg.MODEL.LOSSES.TRI.SCALE,
                },
            },
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        if self.use_hierarchical:
            features_dict = self.backbone(images, return_hierarchical=True)
            features = self.attn(features_dict) if self.attn is not None else features_dict["layer4"]
        else:
            features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN compatibility: avoid invalid class indices in borrowed routines.
            if targets.sum() < 0:
                targets.zero_()

            outputs = self.heads(features, targets)
            return self.losses(outputs, targets)

        return self.heads(features)

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"]
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError(
                "batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs))
            )

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute the loss used in the final HOSCA experiment.
        """
        pred_class_logits = outputs["pred_class_logits"].detach()
        cls_outputs = outputs["cls_outputs"]
        pred_features = outputs["features"]

        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs["loss_names"]

        if "CrossEntropyLoss" in loss_names:
            ce_kwargs = self.loss_kwargs["ce"]
            loss_dict["loss_cls"] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                eps=ce_kwargs.get("eps"),
                alpha=ce_kwargs.get("alpha"),
            ) * ce_kwargs.get("scale")

        if "TripletLoss" in loss_names:
            tri_kwargs = self.loss_kwargs["tri"]
            loss_dict["loss_triplet"] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get("margin"),
                tri_kwargs.get("norm_feat"),
                tri_kwargs.get("hard_mining"),
            ) * tri_kwargs.get("scale")

        return loss_dict
