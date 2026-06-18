# HOSCA

## What Is Included

- `hoscareid/layers/coord_att.py`: coordinate attention used by OS-CA.
- `hoscareid/layers/os_ca_block.py`: full OS-CA block for Stage 4.
- `hoscareid/layers/hierarchical_osca.py`: lightweight Stage-3 OS-CA plus hierarchical gated residual fusion.
- `hoscareid/modeling/backbones/resnet_hierarchical.py`: ResNet backbone that can return both `layer3` and `layer4` features.
- `hoscareid/modeling/meta_arch/baseline.py`: trimmed FastReID baseline path for HOSCA training and inference.
- `configs/Market1501/bagtricks_R50_IBN_HOSCA_GeM.yml`: Market1501 training config.
- `configs/MSMT17/bagtricks_R50_IBN_HOSCA_GeM.yml`: MSMT17 training config.
- `tools/train_net.py`: FastReID training/evaluation entrypoint.
- `reproducibility/hosca_config/README.md`: reproduction notes.

## Paper-Code Mapping

| Paper component | Code |
| --- | --- |
| Omni-Scale Coordinate Attention (OS-CA) | `hoscareid/layers/os_ca_block.py`, `hoscareid/layers/coord_att.py` |
| Stage-aware lightweight OS-CA for Stage 3 | `hoscareid/layers/hierarchical_osca.py` |
| Hierarchical residual fusion with independent gates | `hoscareid/layers/hierarchical_osca.py` |
| Stage-3/Stage-4 feature extraction | `hoscareid/modeling/backbones/resnet_hierarchical.py` |
| GeM head + CE/Triplet training path | `hoscareid/modeling/meta_arch/baseline.py`, `configs/*/bagtricks_R50_IBN_HOSCA_GeM.yml` |

## Usage

1. Start from a FastReID codebase with ResNet-IBN, GeM pooling, CrossEntropyLoss, and TripletLoss support.
2. Copy files from `hoscareid/` into the corresponding `fastreid/` paths in your FastReID codebase.
3. Ensure the FastReID config defaults define:

```python
_C.MODEL.BACKBONE.WITH_HOSCA = False
```

4. Train on Market1501:

```bash
python tools/train_net.py --config-file configs/Market1501/bagtricks_R50_IBN_HOSCA_GeM.yml --num-gpus 1
```

5. Train on MSMT17:

```bash
python tools/train_net.py --config-file configs/MSMT17/bagtricks_R50_IBN_HOSCA_GeM.yml --num-gpus 1
```

6. Evaluate a trained checkpoint:

```bash
python tools/train_net.py --config-file configs/Market1501/bagtricks_R50_IBN_HOSCA_GeM.yml --eval-only MODEL.WEIGHTS path/to/model.pth
```

## Notes

- The config `_BASE_` entries refer to standard FastReID base configs.
