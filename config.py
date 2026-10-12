"""全局配置：路径、超参数等。

各个脚本都会 import 这里的配置，方便统一修改。
"""
import os
import cv2

# 项目根目录（本文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据保存目录：data/<分数 0~10>/xxx.jpg
DATA_DIR = os.path.join(BASE_DIR, "data")

# 模型保存目录与保存路径
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pth")

# 人脸检测用到的 Haar 级联分类器。
# 优先使用项目内自带的 xml 文件（部分 OpenCV 版本不再附带该文件），
# 若项目内没有，则回退到 OpenCV 自带目录。
_CASCADE_FILE = "haarcascade_frontalface_default.xml"
_LOCAL_CASCADE = os.path.join(BASE_DIR, _CASCADE_FILE)
CASCADE_PATH = _LOCAL_CASCADE if os.path.exists(_LOCAL_CASCADE) \
    else cv2.data.haarcascades + _CASCADE_FILE

# 输入给网络的图像尺寸（灰度图，正方形）
IMG_SIZE = 48

# 训练超参数
BATCH_SIZE = 16     # 每批训练的图片数量
EPOCHS = 20         # 训练轮数
LEARNING_RATE = 1e-3  # 学习率
VAL_RATIO = 0.2     # 验证集占全部数据的比例
