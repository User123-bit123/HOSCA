# encoding: utf-8
"""
Minimal Baseline meta-architecture for the HOSCA release pack.

This version only keeps the code path required by the final
`bagtricks_R50_IBN_HOSCA_GeM_v8_map_opt` experiment:

- hierarchical backbone feature extraction
- HierarchicalOSCA attention/fusion
- optional Part-Competitive Learning (PCL) branch
- embedding head forward
- CrossEntropyLoss + TripletLoss
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import cross_entropy_loss, log_accuracy, triplet_loss
from fastreid.layers.hierarchical_osca import HierarchicalOSCA
from fastreid.layers.part_competitive import PartCompetitiveBranch
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Minimal Baseline architecture for the HOSCA release.
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
        hosca_kwargs=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.attn = layers_attn
        self.use_hierarchical = use_hierarchical
        self.loss_kwargs = loss_kwargs or {}
        self.hosca_kwargs = hosca_kwargs or {}
        self.hosca_part_enabled = bool(
            self.hosca_kwargs.get("part_competitive", False)
        )
        self.hosca_part_ce_scale = float(
            self.hosca_kwargs.get("part_ce_scale", 0.5)
        )
        self.hosca_part_tri_scale = float(
            self.hosca_kwargs.get("part_tri_scale", 0.5)
        )
        self.hosca_part_balance_scale = float(
            self.hosca_kwargs.get("part_balance_scale", 0.01)
        )
        self.hosca_part_eval_weight = float(
            self.hosca_kwargs.get("part_eval_weight", 0.75)
        )
        self.hosca_part_gradient_isolation = bool(
            self.hosca_kwargs.get("part_gradient_isolation", False)
        )
        self.hosca_part_init_seed = int(
            self.hosca_kwargs.get("part_init_seed", 0)
        )

        if self.hosca_part_eval_weight < 0:
            raise ValueError("part_eval_weight must be non-negative")
        if self.hosca_part_init_seed < 0:
            raise ValueError("part_init_seed must be non-negative")

        if self.hosca_part_enabled:
            part_branch_kwargs = dict(
                layer3_channels=1024,
                fused_channels=2048,
                num_parts=self.hosca_kwargs.get("part_num", 4),
                part_dim=self.hosca_kwargs.get("part_dim", 256),
                num_classes=self.hosca_kwargs.get("num_classes", 0),
                mask_temperature=self.hosca_kwargs.get(
                    "part_mask_temperature", 1.0
                ),
                vertical_prior_strength=self.hosca_kwargs.get(
                    "part_vertical_prior_strength", 1.0
                ),
                vertical_prior_sigma=self.hosca_kwargs.get(
                    "part_vertical_prior_sigma", 0.18
                ),
            )
            if self.hosca_part_gradient_isolation:
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(self.hosca_part_init_seed)
                    self.part_branch = PartCompetitiveBranch(
                        **part_branch_kwargs
                    )
            else:
                self.part_branch = PartCompetitiveBranch(**part_branch_kwargs)
        else:
            self.part_branch = None

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
            "hosca_kwargs": {
                "part_competitive": cfg.MODEL.HOSCA.PART_COMPETITIVE,
                "part_num": cfg.MODEL.HOSCA.PART_NUM,
                "part_dim": cfg.MODEL.HOSCA.PART_DIM,
                "part_mask_temperature": cfg.MODEL.HOSCA.PART_MASK_TEMPERATURE,
                "part_vertical_prior_strength": cfg.MODEL.HOSCA.PART_VERTICAL_PRIOR_STRENGTH,
                "part_vertical_prior_sigma": cfg.MODEL.HOSCA.PART_VERTICAL_PRIOR_SIGMA,
                "part_ce_scale": cfg.MODEL.HOSCA.PART_CE_SCALE,
                "part_tri_scale": cfg.MODEL.HOSCA.PART_TRI_SCALE,
                "part_balance_scale": cfg.MODEL.HOSCA.PART_BALANCE_SCALE,
                "part_eval_weight": cfg.MODEL.HOSCA.PART_EVAL_WEIGHT,
                "part_gradient_isolation": cfg.MODEL.HOSCA.PART_GRADIENT_ISOLATION,
                "part_init_seed": cfg.MODEL.HOSCA.PART_INIT_SEED,
                "num_classes": cfg.MODEL.HEADS.NUM_CLASSES,
            },
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        hosca_aux = None
        if self.use_hierarchical:
            features_dict = self.backbone(images, return_hierarchical=True)
            if self.attn is not None:
                hosca_result = self.attn(
                    features_dict,
                    return_aux=self.part_branch is not None,
                )
                if isinstance(hosca_result, dict):
                    features = hosca_result["fused"]
                    hosca_aux = hosca_result
                else:
                    features = hosca_result
            else:
                features = features_dict["layer4"]
        else:
            features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # Keep compatibility with borrowed FastReID training utilities.
            if targets.sum() < 0:
                targets.zero_()

            outputs = self.heads(features, targets)
            part_outputs = self._build_part_outputs(hosca_aux)
            return self.losses(outputs, targets, part_outputs=part_outputs)

        outputs = self.heads(features)
        part_outputs = self._build_part_outputs(hosca_aux)
        if part_outputs is None:
            return outputs

        global_features = F.normalize(outputs.float(), dim=1)
        part_features = F.normalize(part_outputs["features"].float(), dim=1)
        return F.normalize(
            torch.cat(
                [global_features, self.hosca_part_eval_weight * part_features],
                dim=1,
            ),
            dim=1,
        )

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

    def _build_part_outputs(self, hosca_aux):
        if self.part_branch is None or hosca_aux is None:
            return None

        layer3_features = hosca_aux.get("layer3")
        fused_features = hosca_aux.get("fused")
        if layer3_features is None or fused_features is None:
            return None

        if self.hosca_part_gradient_isolation:
            layer3_features = layer3_features.detach()
            fused_features = fused_features.detach()
        return self.part_branch(layer3_features, fused_features)

    def losses(self, outputs, gt_labels, part_outputs=None):
        """
        Compute the losses used in the final HOSCA experiment.
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

        if part_outputs is not None:
            ce_kwargs = self.loss_kwargs["ce"]
            tri_kwargs = self.loss_kwargs["tri"]
            if "CrossEntropyLoss" in loss_names:
                loss_dict["loss_part_cls"] = cross_entropy_loss(
                    part_outputs["cls_outputs"],
                    gt_labels,
                    eps=ce_kwargs.get("eps"),
                    alpha=ce_kwargs.get("alpha"),
                ) * ce_kwargs.get("scale") * self.hosca_part_ce_scale
            if "TripletLoss" in loss_names:
                loss_dict["loss_part_triplet"] = triplet_loss(
                    part_outputs["features"],
                    gt_labels,
                    tri_kwargs.get("margin"),
                    tri_kwargs.get("norm_feat"),
                    tri_kwargs.get("hard_mining"),
                ) * tri_kwargs.get("scale") * self.hosca_part_tri_scale
            loss_dict["loss_part_balance"] = (
                part_outputs["balance_loss"] * self.hosca_part_balance_scale
            )

        return loss_dict
