"""训练脚本：读取 data 目录中的图片，训练 CNN，并保存模型到 models/。

训练前请先运行 dataset.py 采集数据，每个分数（0~10）最好各有几十张。
"""
import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config
from model import EmotionNet


class EmotionDataset(Dataset):
    """从 (图片路径, 分数) 列表构建的自定义数据集。"""

    def __init__(self, image_paths, scores, transform=None):
        self.image_paths = image_paths
        self.scores = scores
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:  # 容错：文件损坏时用全黑图代替
            img = np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)
        img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        return img, score


def load_data():
    """扫描 data 目录，返回 (图片路径列表, 分数列表)。"""
    image_paths, scores = [], []
    for score in range(11):
        folder = os.path.join(config.DATA_DIR, str(score))
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(folder, fname))
                scores.append(float(score))
    print(f"共找到 {len(image_paths)} 张图片")
    return image_paths, scores


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    image_paths, scores = load_data()
    if len(image_paths) < 10:
        print("数据太少，请先运行 dataset.py 采集数据！")
        return

    # 打乱数据，并按比例划分训练集 / 验证集
    data = list(zip(image_paths, scores))
    random.shuffle(data)
    n_val = int(len(data) * config.VAL_RATIO)
    val_data = data[:n_val]
    train_data = data[n_val:]

    # 图片预处理：转成 tensor 并归一化到 0~1
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = EmotionDataset([d[0] for d in train_data],
                               [d[1] for d in train_data], transform)
    val_set = EmotionDataset([d[0] for d in val_data],
                             [d[1] for d in val_data], transform)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE)

    model = EmotionNet().to(device)
    criterion = nn.MSELoss()   # 回归任务用均方误差
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        # ---- 训练阶段 ----
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs).squeeze(1)  # (B,) 预测分数
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        train_loss = total_loss / len(train_set)

        # ---- 验证阶段 ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs).squeeze(1)
                val_loss += criterion(outputs, labels).item() * imgs.size(0)
        val_loss /= len(val_set)

        print(f"Epoch {epoch + 1}/{config.EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # 保存模型
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"训练完成，模型已保存到 {config.MODEL_PATH}")


if __name__ == "__main__":
    train()
