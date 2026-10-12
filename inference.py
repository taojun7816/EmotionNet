"""实时推理：打开摄像头，检测人脸并实时显示 0~10 的情绪分数。

请先运行 train.py 训练并保存模型，再运行本脚本。
按 q 退出。
"""
import cv2
import torch
from torchvision import transforms
import config
from model import EmotionNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型
    model = EmotionNet().to(device)
    try:
        model.load_state_dict(
            torch.load(config.MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        print("模型加载成功！")
    except FileNotFoundError:
        print(f"未找到模型 {config.MODEL_PATH}，请先运行 train.py 训练！")
        return

    transform = transforms.Compose([transforms.ToTensor()])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    face_cascade = cv2.CascadeClassifier(config.CASCADE_PATH)
    print("按 q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 裁剪人脸 -> 缩放到 48x48 -> 转 tensor -> 加 batch 维度
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (config.IMG_SIZE, config.IMG_SIZE))
            tensor = transform(face).unsqueeze(0).to(device)  # (1,1,48,48)

            with torch.no_grad():
                score = model(tensor).item()

            # 在框上方显示分数
            cv2.putText(frame, f"Score: {score:.1f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Score (0=most sad, 10=most happy)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
