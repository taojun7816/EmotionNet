# 人脸表情 → 心情分数（0~10）Demo

用简单的 CNN 学习「人脸表情」和「心情分数」的对应关系：
**0 分 = 最伤心，10 分 = 最开心**，中间为连续分数。

## 文件说明

| 文件 | 功能 |
| --- | --- |
| `config.py` | 全局配置：路径、图像尺寸、训练超参数 |
| `dataset.py` | 数据采集 UI：摄像头实时显示 + 键盘打分保存图片 |
| `model.py` | 网络结构设计：简单 CNN（回归输出 0~10） |
| `train.py` | 训练网络并保存模型 |
| `inference.py` | 实时推理 UI：摄像头显示人脸 + 推演分数 |
| `requirements.txt` | 依赖列表 |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用步骤

### 1. 采集数据（`dataset.py`）

```bash
python dataset.py
```

- 数字键 `0~9`：设置分数 0~9
- 字母键 `t`：设置分数 10
- 字母键 `s`：保存当前人脸（保存到 `data/<分数>/`）
- 字母键 `q`：退出

建议每个分数各采集几十张不同表情的人脸，数据越多效果越好。

### 2. 训练（`train.py`）

```bash
python train.py
```

训练完成后模型保存到 `models/emotion_model.pth`。

### 3. 实时推理（`inference.py`）

```bash
python inference.py
```

摄像头画面中检测到的人脸上会实时显示推演出的分数（0~10）。

## 提示

- 本 demo 使用 OpenCV 自带的 Haar 级联做人脸检测，无需额外下载模型。
- 网络输出为回归分数（0~10 连续值），训练用均方误差 MSE。
- 若训练效果不佳，可增加数据量、调整 `config.py` 中的 `EPOCHS` 等超参数。
