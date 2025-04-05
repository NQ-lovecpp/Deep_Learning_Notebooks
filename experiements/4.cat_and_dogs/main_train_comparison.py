import os
import re
import csv
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ======== (1) 导入 torchvision 预训练模型 ========
from torchvision.models import resnet50, mobilenet_v2, efficientnet_b0
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

# ======== (2) 你的自定义网络（示例） ========
from models.ResidualVGGTransformer import ResidualVGGTransformer

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms

class CatDogDataset(Dataset):
    def __init__(self, folder, transform=None):
        """
        folder: 例如 "./dataset/train"
        假设里面混了 cat.***.jpg 和 dog.***.jpg
        """
        self.folder = folder
        self.transform = transform
        
        # 收集所有.jpg文件
        self.image_files = [
            f for f in os.listdir(folder) if f.endswith('.jpg')
        ]
        
        # 这里简单地通过文件名前缀 cat. / dog. 判定label
        # label=0 => cat, label=1 => dog
        self.samples = []
        for fname in self.image_files:
            if fname.startswith('cat'):
                label = 0
            elif fname.startswith('dog'):
                label = 1
            else:
                # 如果有不符合命名的文件，跳过或 raise
                continue
            self.samples.append((fname, label))
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.folder, fname)
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 图像增广/预处理
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(
    data_dir, 
    img_size=150, 
    batch_size=32,
    train_ratio=0.8,
    seed=42
):
    """
    在同一个 data_dir 中混存 cat.***.jpg / dog.***.jpg
    用自定义 Dataset 解析 label
    并根据 train_ratio 分割成 train/val
    """
    # 1) 定义训练/验证集的 transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # 2) 先实例化一个不带 transform 的完整 dataset
    full_dataset = CatDogDataset(folder=data_dir, transform=None)
    
    # 3) 按比例拆分
    full_length = len(full_dataset)
    train_length = int(full_length * train_ratio)
    val_length = full_length - train_length
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_subset, val_subset = random_split(
        full_dataset, [train_length, val_length], generator=g
    )
    
    # 4) 分别给 train_subset, val_subset 套用 transforms
    #    因为 random_split 返回的是 Subset，不支持直接 transform
    #    所以写个包装类
    class TransformWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if self.transform:
                img = self.transform(img)
            return img, label
    
    train_dataset = TransformWrapper(train_subset, train_transform)
    val_dataset   = TransformWrapper(val_subset,   val_transform)
    
    # 5) 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader



# ======== (4) 冻结模型参数（示例） ========
def freeze_backbone(model, final_layer_name: str):
    for name, param in model.named_parameters():
        if final_layer_name not in name:
            param.requires_grad = False

# ======== (5) 训练与验证函数 ========
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

# ======== (6) 训练主流程，保存CSV & 画图 ========
def train_model(model, model_name, train_loader, val_loader, device, 
                num_epochs=50, lr=1e-4, out_dir="./outputs", input_size=150):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{model_name}_logs.csv")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # 写CSV表头
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 写入CSV
        with open(csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, current_lr])

        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}")

    # 画并保存曲线
    plot_path = os.path.join(out_dir, f"{model_name}_curve.png")
    plot_training_curve(train_losses, val_losses, train_accs, val_accs, model_name, plot_path)

    # ======== (6.1) 导出 ONNX 模型 ========
    onnx_path = os.path.join(out_dir, f"{model_name}.onnx")
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    model.eval()
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        opset_version=14,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"[{model_name}] Exported to {onnx_path}")

    return model

def plot_training_curve(train_losses, val_losses, train_accs, val_accs, model_name, save_path):
    """ 绘制并保存单个模型的训练曲线 """
    epochs = list(range(1, len(train_losses)+1))
    
    fig, ax = plt.subplots(2, 1, figsize=(8,8), dpi=100)
    ax[0].plot(epochs, train_losses, label="Train Loss", color="blue")
    ax[0].plot(epochs, val_losses,   label="Val Loss",   color="red")
    ax[0].set_title(f"[{model_name}] Loss Curve")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(epochs, train_accs, label="Train Acc", color="blue")
    ax[1].plot(epochs, val_accs,   label="Val Acc",   color="red")
    ax[1].set_title(f"[{model_name}] Accuracy Curve")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved curve figure to: {save_path}")

# ======== (7) 对比多模型的验证准确率 ========
def plot_all_models_in_one(model_names, out_dir="./outputs"):
    plt.figure(figsize=(8,6))
    for m_name in model_names:
        csv_path = os.path.join(out_dir, f"{m_name}_logs.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skip.")
            continue
        df = pd.read_csv(csv_path)
        val_accs = df["val_acc"].tolist()
        epochs = range(1, len(val_accs)+1)
        plt.plot(epochs, val_accs, label=m_name)
    
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.legend()
    save_path = os.path.join(out_dir, "compare_val_acc.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison figure to: {save_path}")

# ======== (8) 主函数入口 ========
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据集路径（示例）
    Datadir = "./dataset/train"
    out_dir   = "./outputs"
    img_size  = 150
    num_epochs= 50

    # 获得DataLoader
    train_loader, val_loader = get_dataloaders(
        data_dir=Datadir, 
        img_size=img_size, 
        batch_size=32, 
        train_ratio=0.8
    )

    # 1) ResNet50
    net1 = resnet50(weights="IMAGENET1K_V1")
    freeze_backbone(net1, final_layer_name="fc")
    net1.fc = nn.Linear(net1.fc.in_features, 2)
    net1.to(device)
    model_name1 = "ResNet50"

    # 2) MobileNetV2
    net2 = mobilenet_v2(weights="IMAGENET1K_V1")
    for param in net2.features.parameters():
        param.requires_grad = False
    net2.classifier[1] = nn.Linear(net2.classifier[1].in_features, 2)
    net2.to(device)
    model_name2 = "MobileNetV2"

    # 3) EfficientNet-B0
    net3 = efficientnet_b0(weights="IMAGENET1K_V1")
    for param in net3.features.parameters():
        param.requires_grad = False
    net3.classifier[1] = nn.Linear(net3.classifier[1].in_features, 2)
    net3.to(device)
    model_name3 = "EfficientNetB0"

    # 4) 我的自定义网络 
    net4 = ResidualVGGTransformer(num_classes=2)
    net4.to(device)
    model_name4 = "ResidualVGGTransformer"

    # 分别训练 + 导出 ONNX
    train_model(net1, model_name1, train_loader, val_loader, device, num_epochs=num_epochs, lr=1e-4, out_dir=out_dir, input_size=img_size)
    train_model(net2, model_name2, train_loader, val_loader, device, num_epochs=num_epochs, lr=1e-4, out_dir=out_dir, input_size=img_size)
    train_model(net3, model_name3, train_loader, val_loader, device, num_epochs=num_epochs, lr=1e-4, out_dir=out_dir, input_size=img_size)
    train_model(net4, model_name4, train_loader, val_loader, device, num_epochs=num_epochs, lr=1e-4, out_dir=out_dir, input_size=img_size)

    # 对比图
    model_names = [model_name1, model_name2, model_name3, model_name4]
    plot_all_models_in_one(model_names, out_dir=out_dir)

    print("All tasks done!")
