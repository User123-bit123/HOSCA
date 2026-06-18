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
