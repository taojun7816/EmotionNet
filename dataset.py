"""数据采集：打开摄像头实时显示，用户按键打分并保存人脸图片。

按键说明：
  数字键 0~9 : 设置当前分数为 0~9
  字母键 t   : 设置当前分数为 10
  字母键 s   : 保存当前检测到的人脸（以当前分数保存）
  字母键 q   : 退出

保存目录结构：
  data/0/、data/1/ ... data/10/，每个文件夹里是该分数的人脸图片。
"""
import os
import time
import cv2
import config


def main():
    # 创建 0~10 共 11 个保存文件夹
    for score in range(11):
        os.makedirs(os.path.join(config.DATA_DIR, str(score)), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    face_cascade = cv2.CascadeClassifier(config.CASCADE_PATH)
    current_score = 5   # 默认分数
    saved_count = 0     # 已保存图片数量

    print("=" * 50)
    print("按键说明：")
    print("  数字键 0~9 : 设置分数 0~9")
    print("  字母键 t   : 设置分数 10")
    print("  字母键 s   : 保存当前人脸图片")
    print("  字母键 q   : 退出")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # 在画面上框出人脸
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 在画面上显示当前分数
        cv2.putText(frame, f"Score: {current_score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "0-9:set  t:10  s:save  q:quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('9'):
            current_score = key - ord('0')
            print(f"当前分数设置为: {current_score}")
        elif key == ord('t'):
            current_score = 10
            print("当前分数设置为: 10")
        elif key == ord('s'):
            if len(faces) == 0:
                print("未检测到人脸，请把脸对准摄像头！")
            else:
                # 只保存第一张检测到的人脸
                x, y, w, h = faces[0]
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (config.IMG_SIZE, config.IMG_SIZE))
                fname = f"{int(time.time() * 1000)}.jpg"
                path = os.path.join(config.DATA_DIR, str(current_score), fname)
                cv2.imwrite(path, face)
                saved_count += 1
                print(f"已保存第 {saved_count} 张：分数={current_score} -> {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"采集结束，共保存 {saved_count} 张图片。")


if __name__ == "__main__":
    main()
