#train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import  transforms
from torchvision.models import resnet34, ResNet34_Weights
from dataset import ChestXrayDataset  
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='data/raw')
parser.add_argument('--output_dir', type=str, default='data/processed')
parser.add_argument('--mode', type=str, default='none',
                    choices=['none', 'spatial', 'frequency'],  
                    help='预处理模式: none-原始基准, spatial-空间域去噪, frequency-频域去噪')
parser.add_argument('--data_dir', type=str, default='data/processed_none')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gan_path', type=str, default='pretrained/denoise_gan.pth',
                    help='Path to pretrained GAN model')
args = parser.parse_args()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return epoch_loss, epoch_acc, precision, recall, f1


if __name__ == '__main__':

    device = torch.device('cuda')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],
                             std=[0.229])
    ])

    dataset = ChestXrayDataset(args.data_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)

    pretrained_conv1 = model.conv1.weight.data
    new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv1.weight = nn.Parameter(pretrained_conv1.mean(dim=1, keepdim=True))
    model.conv1 = new_conv1

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, precision, recall, f1 = validate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print(f'Val   Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet34.pth')
            print('保存最优模型\n')
