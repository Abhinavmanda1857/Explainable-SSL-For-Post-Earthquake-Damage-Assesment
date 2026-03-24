## Checkpoints

### This folder contains the saved model weights used for inference and evaluation.

### Included:

 - vit_tiny_epoch03_miou0.6246.pth — Vision Transformer (ViT-Tiny) supervised baseline model trained for 4-class damage segmentation.

## Notes:

 - These checkpoints are loaded automatically by the Streamlit app (app.py) when providing the correct path.

 - Use these weights for inference or further fine-tuning.

 - Do not upload large checkpoints to GitHub if the file size exceeds repository limits—use Git LFS instead
