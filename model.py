import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score

class UNetEncoder(nn.Module):
    """轻量级U-Net编码器作为特征提取器"""
    def __init__(self, in_channels=1, init_features=32):
        super().__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features*2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features*2, features*4, name="enc3")
        
        self.attention = nn.Sequential(
            nn.Conv2d(features*4, features*4//4, kernel_size=1),  
            nn.ReLU(),
            nn.Conv2d(features*4//4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)       # [B,32,H,W]
        enc2 = self.encoder2(self.pool1(enc1))  # [B,64,H/2,W/2]
        enc3 = self.encoder3(self.pool2(enc2))  # [B,128,H/4,W/4]
        
        attn_map = self.attention(enc3)
        return enc3 * attn_map

class UNetResNet(nn.Module):
    """组合U-Net特征提取器与ResNet34"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.unet_encoder = UNetEncoder()
        self.resnet = self._build_adapted_resnet()
        self.fc = nn.Linear(512, num_classes)

    def _build_adapted_resnet(self):
        model = resnet34(weights=True)
        
        original_conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            128, 64,  # 输入通道改为UNet输出维度
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        with torch.no_grad():
            model.conv1.weight[:,:] = original_conv1.weight.mean(dim=1, keepdim=True)
        
        model.fc = nn.Identity()
        return model

    def forward(self, x):
        unet_features = self.unet_encoder(x)  # [B,128,H/4,W/4]
        features = self.resnet(unet_features)  # [B,512,1,1]
        
        return self.fc(features.view(features.size(0), -1))

def visualize_features(features, epoch, batch_idx):
    os.makedirs('features', exist_ok=True)
    plt.figure(figsize=(12,6))
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(features[0, i*32].cpu().detach(), cmap='jet')
        plt.axis('off')
    plt.savefig(f'features/epoch{epoch}_batch{batch_idx}.png')
    plt.close()

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 100 == 0:
            with torch.no_grad():
                features = model.unet_encoder(images[:1])
                visualize_features(features, epoch, batch_idx)

    return running_loss / total, correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    metrics = {
        'loss': running_loss / total,
        'acc': correct / total,
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }
    return metrics

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return loss.mean()
