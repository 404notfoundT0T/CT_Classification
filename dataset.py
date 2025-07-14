#dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from preprocess import apply_preprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='data/raw')
parser.add_argument('--output_dir', type=str, default='data/processed')
parser.add_argument('--mode', type=str, default='none',
                    choices=['none', 'spatial', 'frequency'],  
                    help='预处理模式: none-原始基准, spatial-空间域去噪, frequency-频域去噪')
parser.add_argument('--data_dir', type=str, default='data/processed_none')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None,mode=args.mode):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  
        self.mode = mode

        for label, subdir in enumerate(['Normal', 'Abnormal']):
            subdir_path = os.path.join(root_dir, subdir)
            if not os.path.exists(subdir_path):
                continue
            for fname in os.listdir(subdir_path):
                if fname.lower().endswith('.png'):
                    self.samples.append((os.path.join(subdir_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = apply_preprocessing(img, mode=self.mode)  

        if self.transform:
            img = self.transform(Image.fromarray(img)) 

        return img, label
