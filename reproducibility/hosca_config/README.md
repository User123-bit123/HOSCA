# HOSCA Reproducibility Notes

This folder records compact reproduction settings for the HOSCA configuration.

## Market1501 Run

- Config: `configs/Market1501/bagtricks_R50_IBN_HOSCA_GeM.yml`
- Backbone: ResNet-50-IBN with hierarchical Stage-3/Stage-4 outputs
- Attention: HOSCA enabled
- Pooling: Generalized Mean Pooling
- Losses: CrossEntropyLoss with label smoothing and TripletLoss
- Input size: 256 x 128
- Epochs: 120
- Batch size: 64
- Evaluation: no flip test-time augmentation, no re-ranking

## HOSCA+PCL Run

- Config: `configs/Market1501/bagtricks_R50_IBN_HOSCA_PCL.yml` or the MSMT17 counterpart
- PCL: 4 competing soft parts, 256 dimensions per part
- PCL losses: classification, triplet, and part-balance supervision
- Inference: normalized global-local concatenation with `PART_EVAL_WEIGHT=0.75`
- Backbone stride: `LAST_STRIDE=1`; Stage 3 and Stage 4 are aligned at `H/16 x W/16`
- Configuration fields: see `pcl_defaults.py`
