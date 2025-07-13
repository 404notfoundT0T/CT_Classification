import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 预处理后数据根目录，结构：
            root_dir/
                normal/
                abnormal/
        transform: torchvision.transforms 用于图像增强或预处理
        """
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []  

        for label, subdir in enumerate(['normal', 'abnormal']):
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
        image = Image.open(img_path).convert('L')  

        if self.transform:
            image = self.transform(image)

        return image, label
