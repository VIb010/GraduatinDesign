import os
import glob
from PIL import Image
import torch
import torch.nn as nn
# === 修改：换用 AdamW ===
from torch.optim import AdamW
# === 新增：导入调度器 ===
from torch.optim.lr_scheduler import ReduceLROnPlateau
# === end ===
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import YOLODetectionDataset

def train_model(model, dataloaders, criterion, optimizer, scheduler, device,
                num_epochs=50, patience=5, checkpoint_dir="checkpoints"):
    """
    训练并验证模型，并集成 ReduceLROnPlateau 调度与 EarlyStopping。
    """
    writer = SummaryWriter(log_dir="./runs")
    best_acc = 0.0
    epochs_no_improve = 0
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc  = running_corrects.double() / len(dataloaders['train'].dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(dataloaders['val'].dataset)
        epoch_val_acc  = val_corrects.double() / len(dataloaders['val'].dataset)
        print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        writer.add_scalar("Loss/val", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/val", epoch_val_acc, epoch)

        # === 修改：ReduceLROnPlateau 调度 ===
        scheduler.step(epoch_val_acc)
        # === end ===

        # 检查点 & EarlyStopping
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            epochs_no_improve = 0
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} epochs, early stopping.")
                break

        print()

    writer.close()
    return model

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(dataloader.dataset)
    total_acc  = running_corrects.double() / len(dataloader.dataset)
    return total_loss, total_acc

def main():
    data_dir = "./Acne"

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),           # === 新增轻度数据增强 ===
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # === 新增 ===
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ]),
    }

    # 使用 YOLODetectionDataset 加载
    train_dataset = YOLODetectionDataset(
        images_dir=os.path.join(data_dir, 'train/images'),
        labels_dir=os.path.join(data_dir, 'train/labels'),
        transform = data_transforms['train']
    )
    val_dataset = YOLODetectionDataset(
        images_dir=os.path.join(data_dir, 'val/images'),
        labels_dir=os.path.join(data_dir, 'val/labels'),
        transform = data_transforms['val']
    )
    test_dataset = YOLODetectionDataset(
        images_dir=os.path.join(data_dir, 'test/images'),
        labels_dir=os.path.join(data_dir, 'test/labels'),
        transform = data_transforms['test']
    )

    # 调试打印（可注释）
    # print("Train samples:", len(train_dataset))
    # print("Val   samples:", len(val_dataset))
    # print("Test  samples:", len(test_dataset))

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4),
        'val':   DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=4),
    }
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 构建模型
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    num_classes = max([s[1] for s in train_dataset.samples]) + 1
    print(f"检测到 {num_classes} 个类别")
    # === 新增：在全连接前加 Dropout ===
    model_ft.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    model_ft = model_ft.to(device)

    # 损失与优化器
    criterion = nn.CrossEntropyLoss()
    # === 修改：换 AdamW 并加权衰减 ===
    optimizer_ft = AdamW(model_ft.parameters(), lr=1e-3, weight_decay=1e-4)
    # === 新增：ReduceLROnPlateau 调度 ===
    scheduler = ReduceLROnPlateau(optimizer_ft, mode='max', factor=0.1, patience=3)
    # === end ===

    # 训练 & 验证（带 EarlyStopping，最大 50 epochs，patience=5）
    trained_model = train_model(
        model_ft, dataloaders, criterion,
        optimizer_ft, scheduler,
        device,
        num_epochs=50,     # === 修改：最大训练轮数 50 ===
        patience=5         # === 修改：EarlyStopping 耐心值 5 ===
    )

    # 保存最终模型
    torch.save(trained_model.state_dict(), "final_model.pth")
    print("最终模型已保存为 final_model.pth")

    # 在测试集上评估
    test_loss, test_acc = evaluate_model(trained_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
