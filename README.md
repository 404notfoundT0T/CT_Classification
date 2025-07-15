# BayR34-- a Chest X-ray Image Classification Model

This project is designed for `​​ECE 4513 (Image Processing and Computer Vision)`​​ and implements a ​​binary classification system​​ to distinguish between ​​normal and abnormal chest CT/X-ray images​​ using ​​PyTorch​​ and ​​transfer learning​​ with ​​ResNet34​​.

## 📂 Dataset

This project utilizes the **Shenzhen Chest X-ray Set**, a publicly available dataset designed for tuberculosis diagnosis using chest radiographs.

### 📌 Dataset Overview

The Shenzhen Chest X-ray Set is a tuberculosis digital imaging dataset created by the **Lister Hill National Center for Biomedical Communications (LHNCBC)** at the **U.S. National Library of Medicine (NLM)** in collaboration with the **Third People's Hospital of Shenzhen** and **Guangdong Medical College** in China.

- 📸 **Total Images**: 662 chest X-ray images  
- 🧍 **Normal Cases**: 326  
- ⚠️ **Abnormal (TB) Cases**: 336  
- 🖼️ **Image Format**: PNG  
- 📐 **Resolution**: Up to 3000×3000  
- 📄 **Clinical Info**: Accompanying `.txt` files include age, gender, and diagnostic remarks  

> This dataset has been de-identified and is exempt from IRB (Institutional Review Board) review.

---

### 📈 Meta Statistics

| Property           | Value                                      |
|--------------------|--------------------------------------------|
| Total samples      | 662                                        |
| Abnormal rate      | 336 / 662 ≈ 50.75%                         |
| Image resolution   | Min: (1130, 948); Max: (3001, 3001); Median: ~2730×2940 |
| Format             | PNG images + TXT clinical data             |
| Size               | ~3.6 GB                                    |

---

### 🔗 Dataset Links

- 🔹 **Official Site**: [LHNCBC TB Image Data Sets](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets)  
- 🔹 **Download ZIP**: [ChinaSet_AllFiles.zip](https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip)  
- 🔹 **Related Publication**: [NIH Article on PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/)



## Image Preprocessing

We provide three specialized preprocessing methods for medical images:

### 0. None (Baseline)
- **Method**: Raw images with only resizing
- **Characteristics**:
  - Preserves original image data
  - Fastest processing
  - Establishes baseline performance

### 1. Spatial Domain Processing 
- **Method**: Bilateral filtering
- **Key Features**:
  - Edge-preserving noise reduction
  - Maintains structural boundaries
  - Computationally efficient

### 2. Frequency Domain Processing
- **Method**: Wavelet threshold denoising
- **Key Features**:
  - Multi-scale noise removal
  - Adaptive thresholding
  - Preserves fine details

### 3. Bayesian Wavelet Denoising
- **Method**: Adaptive wavelet thresholding with Bayesian estimation
- **Key Features**:
- Automatic threshold calculation（Uses Bayesian statistics to determine optimal thresholds）
- Multi-scale processing（Default 4-level wavelet decomposition）
- Adaptive denoising （Level-specific soft-thresholding）
- **Recommended wavelets**: 
  - `bior3.3` (Biorthogonal 3.3)
  - `sym4` (Symlet 4)

#### Technical Specifications
| Parameter        | Default Value | Description                          |
|------------------|---------------|--------------------------------------|
| `wavelet`        | `bior3.3`     | Wavelet basis function               |
| `level`          | 4             | Decomposition levels                 |
| `threshold_mode` | `soft`        | Thresholding method (soft/hard)      |



## Classification Model Architecture

### Core Components
- **Backbone**: Pretrained ResNet-34 model
- **Adaptations**:
  - Modified input layer for grayscale medical images
  - Custom binary classification head (Normal/Abnormal)
  
### Training Configuration
- **Optimization**: Adam optimizer (1e-4 learning rate)
- **Regularization**: Cross-entropy loss with weight decay
- **Augmentations**: Random flips and normalization

## Run The Model

### 1. Preprocess images

```bash
python preprocess_images.py --input_dir data/raw --output_dir data/processed_clahe --mode clahe
```
Available modes: `bayesian`,`spatial`, `frequency`,` none`

### 2. Train the model

```bash
python train.py --data_dir data/processed_clahe --batch_size 32 --epochs 10 --lr 0.0001
```
### Evaluation Metrics
- Training and validation loss and accuracy
- Validation precision, recall, and F1-score

## Environment Dependencies
- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- OpenCV (cv2)
- Pillow (PIL)

## Project Structure
```bash
├── dataset.py              # Dataset class
├── preprocess_images.py    # Image preprocessing script
├── train.py                # Training script
├── data/                   # Data directory
│   ├── raw/                # Raw images
│   └── processed_X/    # Processed images
```
## Contact
- Feel Free to contact the author if you have any questions.
  
