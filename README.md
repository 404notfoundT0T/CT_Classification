# CT Classification

This project performs binary classification (normal vs abnormal) on chest CT (or X-ray) images using PyTorch and transfer learning with ResNet34.

## Dataset

- The dataset consists of chest X-ray images divided into two classes:
  - `normal`
  - `abnormal`
- Example directory structure:
  ├── dataset.py              # Dataset class
├── preprocess_images.py    # Image preprocessing script
├── train.py                # Training script
├── data/                   # Data directory
│   ├── raw/                # Raw images
│   └── processed_clahe/    # Processed images

