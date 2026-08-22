# HOSCA

## What Is Included

- `hoscareid/layers/coord_att.py`: coordinate attention used by OS-CA.
- `hoscareid/layers/os_ca_block.py`: full OS-CA block for Stage 4.
- `hoscareid/layers/hierarchical_osca.py`: lightweight Stage-3 OS-CA plus hierarchical gated residual fusion.
- `hoscareid/layers/part_competitive.py`: PCL masks, masked pooling, reliability fusion, and local descriptor head.
- `hoscareid/modeling/backbones/resnet_hierarchical.py`: ResNet backbone that can return both `layer3` and `layer4` features.
- `hoscareid/modeling/meta_arch/baseline.py`: trimmed FastReID baseline path for HOSCA/PCL training and inference.
- `configs/Market1501/bagtricks_R50_IBN_HOSCA_GeM.yml`: Market1501 training config.
- `configs/Market1501/bagtricks_R50_IBN_HOSCA_PCL.yml`: Market1501 HOSCA+PCL config.
- `configs/MSMT17/bagtricks_R50_IBN_HOSCA_GeM.yml`: MSMT17 training config.
- `configs/MSMT17/bagtricks_R50_IBN_HOSCA_PCL.yml`: MSMT17 HOSCA+PCL config.
- `tools/train_net.py`: FastReID training/evaluation entrypoint.
- `reproducibility/hosca_config/README.md`: reproduction notes.

## Paper-Code Mapping

| Paper component | Code |
| --- | --- |
| Omni-Scale Coordinate Attention (OS-CA) | `hoscareid/layers/os_ca_block.py`, `hoscareid/layers/coord_att.py` |
| Stage-aware lightweight OS-CA for Stage 3 | `hoscareid/layers/hierarchical_osca.py` |
| Hierarchical residual fusion with independent gates | `hoscareid/layers/hierarchical_osca.py` |
| Stage-3/Stage-4 feature extraction | `hoscareid/modeling/backbones/resnet_hierarchical.py` |
| Part-Competitive Learning (PCL) | `hoscareid/layers/part_competitive.py`, `hoscareid/modeling/meta_arch/baseline.py` |
| GeM head + global/local descriptor fusion | `hoscareid/modeling/meta_arch/baseline.py` |
| CE/Triplet training path | `hoscareid/modeling/meta_arch/baseline.py`, `configs/*/bagtricks_R50_IBN_HOSCA_PCL.yml` |

## Usage

1. Start from a FastReID codebase with ResNet-IBN, GeM pooling, CrossEntropyLoss, and TripletLoss support.
2. Copy files from `hoscareid/` into the corresponding `fastreid/` paths in your FastReID codebase.
3. Ensure the FastReID config defaults define:

```python
_C.MODEL.BACKBONE.WITH_HOSCA = False
```

For HOSCA+PCL, also add the fields in
`reproducibility/hosca_config/pcl_defaults.py` to the host FastReID
configuration defaults. The PCL branch is disabled by default and is enabled
by the `*_HOSCA_PCL.yml` files.

4. Train on Market1501:

```bash
python tools/train_net.py --config-file configs/Market1501/bagtricks_R50_IBN_HOSCA_GeM.yml --num-gpus 1
```

5. Train on MSMT17:

```bash
python tools/train_net.py --config-file configs/MSMT17/bagtricks_R50_IBN_HOSCA_GeM.yml --num-gpus 1
```

6. Train the final HOSCA+PCL model on Market1501:

```bash
python tools/train_net.py --config-file configs/Market1501/bagtricks_R50_IBN_HOSCA_PCL.yml --num-gpus 1
```

7. Train the final HOSCA+PCL model on MSMT17:

```bash
python tools/train_net.py --config-file configs/MSMT17/bagtricks_R50_IBN_HOSCA_PCL.yml --num-gpus 1
```

8. Evaluate a trained checkpoint:

```bash
python tools/train_net.py --config-file configs/Market1501/bagtricks_R50_IBN_HOSCA_PCL.yml --eval-only MODEL.WEIGHTS path/to/model.pth
```

## Notes

- The config `_BASE_` entries refer to standard FastReID base configs.
- The final configuration uses `LAST_STRIDE=1`, so Stage 3 and Stage 4 are
  spatially aligned at the `H/16 x W/16` resolution.
- During inference, HOSCA produces a 2048-dimensional global descriptor and
  PCL produces a 1024-dimensional local descriptor. They are normalized and
  concatenated with `PART_EVAL_WEIGHT=0.75` to form the final descriptor.
