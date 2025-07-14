import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from preprocess import apply_preprocessing

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='none'):
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
