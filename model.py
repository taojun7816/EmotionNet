"""网络结构设计：一个简单的 CNN（卷积神经网络）。

任务是一个回归问题：输入一张人脸灰度图，输出 0~10 的连续分数。
0 分 = 最伤心，10 分 = 最开心。
"""
import torch
import torch.nn as nn


class EmotionNet(nn.Module):
    """输入：(N, 1, 48, 48) 的单通道灰度图
       输出：(N, 1) 的分数，范围 0~10
    """

    def __init__(self):
        super().__init__()
        # 特征提取：3 个「卷积 + ReLU + 池化」块，逐步提取表情特征
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 输出 48x48x32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 24x24x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 24x24x64
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 12x12x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 12x12x128
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 6x6x128
        )

        # 回归头：把特征映射成一个 0~1 的分数，最后乘 10 得到 0~10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),   # 防止过拟合
            nn.Linear(256, 1),
            nn.Sigmoid(),      # 输出 0~1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x * 10.0        # 映射到 0~10


if __name__ == "__main__":
    # 简单自测：随机造一个输入，看输出形状和数值
    model = EmotionNet()
    x = torch.randn(1, 1, 48, 48)  # 模拟 1 张 48x48 的灰度图
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出分数: {y.item():.2f}  (范围 0~10)")
