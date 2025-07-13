# CT Classification

This project performs binary classification (normal vs abnormal) on chest CT (or X-ray) images using PyTorch and transfer learning with ResNet34.

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


## Model Construction
### Image Preprocessing

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

### Classification Model

- Uses torchvision’s pretrained `resnet34` model
- Modified first convolutional layer to accept single-channel grayscale images
- Output layer adjusted for 2-class classification

### Run The Model

#### 1. Preprocess images

```bash
python preprocess_images.py --input_dir data/raw --output_dir data/processed_clahe --mode clahe
```
Available modes: `clahe`, `gaussian`, `sharpen`, `antialias`,` none`

#### 2. Train the model

```bash
python train.py --data_dir data/processed_clahe --batch_size 32 --epochs 10 --lr 0.0001
```
### Evaluation Metrics
- Training and validation loss and accuracy
- Validation precision, recall, and F1-score

### Model Dependencies
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
│   └── processed_clahe/    # Processed images
```
## Contact
- Please contact the author if you have any questions.
  
