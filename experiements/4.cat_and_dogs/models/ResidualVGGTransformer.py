#Building model computational graph
import graphviz
import torchviz
import torch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# 1) 定义残差卷积块 (VGG风格 + 残差)
# ----------------------
class ResidualConvBlock(nn.Module):
    """
    包含2个 3×3 卷积(可以加BN和ReLU)，并带跳跃连接的残差结构。
    输入输出通道数相同，若不一致可用1×1卷积做match。
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(in_channels)
        
        # 如果通道不变，这里就不用额外的1x1
        # self.match_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        # shortcut for residual
        residual = x

        # 第1次卷积
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # 第2次卷积
        out = self.bn2(self.conv2(out))
        
        # 残差叠加
        out += residual
        out = F.relu(out, inplace=True)
        return out

# ----------------------
# 2) 主网络: CNN + Transformer
# ----------------------
class ResidualVGGTransformer(nn.Module):
    def __init__(
        self,
        img_size=150,
        num_classes=2,
        d_model=128,
        nhead=8,
        num_transformer_layers=2,
    ):
        """
        - img_size: 输入图像尺寸(正方形假设)
        - num_classes: 分类数 (猫狗=2)
        - d_model: Transformer输入/输出特征维度
        - nhead: 多头注意力头数
        - num_transformer_layers: TransformerEncoder层数
        """
        super().__init__()
        # =========== CNN部分 (VGG 小卷积 + Residual) ===========
        # 输入 (B,3,150,150)
        self.conv_in = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 先把通道提到32
        
        # Block1: 通道=32
        self.block1 = ResidualConvBlock(32)
        self.pool1  = nn.MaxPool2d(2, 2)  # 150 -> 75
        
        # Block2: 通道=32
        self.block2 = ResidualConvBlock(32)
        self.pool2  = nn.MaxPool2d(2, 2)  # 75 -> 37
        
        # Block3: 升通道到64(可选), 或仍然32看你
        self.conv_ch = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 升通道
        self.block3  = ResidualConvBlock(64)
        self.pool3   = nn.MaxPool2d(2, 2)  # 37 -> 18

        # 这里输出 shape = (B,64,18,18)，可人工确认
        # Flatten 前：C=64, H=18, W=18 => 324 tokens, each 64-dim

        # =========== Transformer部分 ===========
        self.d_model = d_model
        # 线性投影: 64 -> d_model
        self.linear_proj = nn.Linear(64, d_model)

        # 位置编码: 大小= 18*18=324; shape=(1, 324, d_model)
        max_seq_len = 18 * 18
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # 定义 TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=4*d_model,  # 前馈层宽度可调
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # =========== 最终分类层 ===========
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B,3,150,150)
        # -- CNN stage --
        out: torch.Tensor = F.relu(self.conv_in(x))   # (B,32,150,150)
        
        out = self.block1(out)         # (B,32,150,150) 残差
        out = self.pool1(out)          # (B,32,75,75)
        
        out = self.block2(out)         # (B,32,75,75)
        out = self.pool2(out)          # (B,32,37,37)

        out = F.relu(self.conv_ch(out))# (B,64,37,37)
        out = self.block3(out)         # (B,64,37,37)
        out = self.pool3(out)          # (B,64,18,18)

        # -- Flatten => (B, 324, 64)
        B, C, H, W = out.shape   # e.g. (B,64,18,18)
        seq_len = H*W            # 324
        out = out.view(B, C, seq_len)          # (B, 64, 324)
        out = out.transpose(1, 2)              # (B, 324, 64)

        # 投影到 d_model
        out = self.linear_proj(out)            # (B, 324, d_model)

        # 加位置编码
        out = out + self.pos_embedding[:, :seq_len, :]

        # -- Transformer Encoder --
        out = self.transformer(out)  # (B, 324, d_model)

        # 全局平均池化 (也可用 out[:,0,:] if 你想自定义 [CLS] token)
        out = out.mean(dim=1)  # (B, d_model)

        # -- 分类 --
        logits = self.classifier(out)  # (B, num_classes)
        return logits




