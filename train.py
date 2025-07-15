import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ChestXrayDataset
from model import UNetResNet , train_one_epoch, validate, FocalLoss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed_none')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    dataset = ChestXrayDataset(args.data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = UNetResNet().to(device)
    criterion = FocalLoss(gamma=1.5,alpha=0.8).to(device)  # 使用Focal Loss
    
    optimizer = torch.optim.Adam(
        model.parameters(),  # 所有参数统一处理
        lr=args.lr,         # 统一学习率
        weight_decay=args.weight_decay  # 统一权重衰减
    )

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)

        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val   Loss: {val_metrics["loss"]:.4f} Acc: {val_metrics["acc"]:.4f}')
        print(f'Val   Precision: {val_metrics["precision"]:.4f} Recall: {val_metrics["recall"]:.4f} F1: {val_metrics["f1"]:.4f}')

    
