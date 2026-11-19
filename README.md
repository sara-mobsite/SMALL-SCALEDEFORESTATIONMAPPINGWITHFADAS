## Pretrained Models

Pretrained checkpoints for the four reconstruction backbones evaluated in FADAS
are provided:

- FADAS (U-Net)
- FADAS (PSPNet)
- FADAS (SegFormer)
- FADAS (DeepLabV3+)

Each checkpoint corresponds to the model state achieving the lowest validation
reconstruction error during training.

The checkpoints can be used directly to reproduce reconstruction-error maps and
the subsequent deforestation-detection experiments without retraining the
models.

The best model weights are available at:

https://huggingface.co/datasets/saramobsite/FADAS/tree/main
