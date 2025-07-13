# CT Classification

This project performs binary classification (normal vs abnormal) on chest CT (or X-ray) images using PyTorch and transfer learning with ResNet34.

## Dataset

- The dataset consists of chest X-ray images divided into two classes:
  - `normal`
  - `abnormal`

## Image Preprocessing

- Supports multiple preprocessing modes, including:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian blur
- Sharpening
- Antialiasing resize
- None (no preprocessing)

- Preprocessing script: `preprocess_images.py`
- Takes raw images as input
- Applies the selected preprocessing method
- Outputs processed images used for training and validation

## Classification Model

- Uses torchvision’s pretrained `resnet34` model
- Modified first convolutional layer to accept single-channel grayscale images
- Output layer adjusted for 2-class classification

## Usage

### 1. Preprocess images

'''bash
python preprocess_images.py --input_dir data/raw --output_dir data/processed_clahe --mode clahe
'''

