"""生成项目文档用到的示意图：
1. images/network_architecture.png —— CNN 网络结构图
2. images/score_scale.png        —— 0~10 心情分数标尺图

用法：在项目根目录执行  python docs/generate_images.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT_DIR, exist_ok=True)


def draw_network():
    """绘制 EmotionNet 网络结构示意图。"""
    # (文字, 颜色)，从上到下依次是网络各层
    layers = [
        ("Input\n1 x 48 x 48", "gainsboro"),
        ("Conv2d  1->32, k=3, p=1\n48 x 48 x 32", "cornflowerblue"),
        ("ReLU", "whitesmoke"),
        ("MaxPool 2x2\n24 x 24 x 32", "orange"),
        ("Conv2d  32->64, k=3, p=1\n24 x 24 x 64", "cornflowerblue"),
        ("ReLU", "whitesmoke"),
        ("MaxPool 2x2\n12 x 12 x 64", "orange"),
        ("Conv2d  64->128, k=3, p=1\n12 x 12 x 128", "cornflowerblue"),
        ("ReLU", "whitesmoke"),
        ("MaxPool 2x2\n6 x 6 x 128", "orange"),
        ("Flatten\n4608", "teal"),
        ("Linear  4608 -> 256", "plum"),
        ("ReLU + Dropout(0.5)", "whitesmoke"),
        ("Linear  256 -> 1", "plum"),
        ("Sigmoid  0~1", "salmon"),
        ("x 10\nScore 0~10", "lightgreen"),
    ]
    n = len(layers)
    fig, ax = plt.subplots(figsize=(6.5, 0.95 * n))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n)
    ax.axis("off")

    for i, (text, color) in enumerate(layers):
        cy = n - 0.5 - i  # 当前层的中心 y 坐标
        ax.add_patch(mpatches.FancyBboxPatch(
            (2, cy - 0.38), 6, 0.76,
            boxstyle="round,pad=0.08,rounding_size=0.1",
            facecolor=color, edgecolor="black", linewidth=1))
        ax.text(5, cy, text, ha="center", va="center", fontsize=9)
        if i < n - 1:
            # 从当前层底部指向下一层顶部
            ax.annotate("", xy=(5, cy - 0.38), xytext=(5, cy - 0.62),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # 左侧分组标签：特征提取 / 回归头
    ax.text(1.0, 10.0, "Feature\nExtraction", ha="center", va="center",
            fontsize=10, color="dimgray", rotation=90)
    ax.text(1.0, 2.5, "Regression\nHead", ha="center", va="center",
            fontsize=10, color="dimgray", rotation=90)

    ax.set_title("EmotionNet Architecture", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "network_architecture.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", out)


def draw_score_scale():
    """绘制 0~10 心情分数标尺：从伤心（0）到开心（10）的表情变化。"""
    fig, axes = plt.subplots(1, 11, figsize=(16, 2.3))
    cmap = plt.get_cmap("coolwarm")

    for s in range(11):
        t = (s - 5) / 5.0  # -1（最伤心） ~ +1（最开心）
        ax = axes[s]
        color = cmap((t + 1) / 2.0)  # 蓝 -> 红

        # 脸（圆）
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=color, lw=2.5))
        # 眼睛
        ax.add_patch(plt.Circle((-0.35, 0.28), 0.09, color="black"))
        ax.add_patch(plt.Circle((0.35, 0.28), 0.09, color="black"))
        # 嘴巴：抛物线，t>0 微笑（∪），t<0 伤心（∩）
        x = np.linspace(-0.5, 0.5, 100)
        a = t * 0.9
        y = a * x ** 2 - 0.45
        ax.plot(x, y, color="black", lw=3)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0, -1.15, str(s), ha="center", va="center", fontsize=13,
                color=color, fontweight="bold")

    axes[0].set_title("Sad", fontsize=12, color="royalblue")
    axes[10].set_title("Happy", fontsize=12, color="firebrick")
    fig.suptitle("Mood Score Scale (0 = most sad, 10 = most happy)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "score_scale.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    draw_network()
    draw_score_scale()
    print("全部示意图生成完成。")
