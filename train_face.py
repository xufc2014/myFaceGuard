import cv2
import os
import numpy as np

# 1. 修正路径：加 r，分隔正确
folder_path = r"C:\Users\xufc\Desktop\FaceGuard\my_face"
# 模型保存路径也用 r
model_path = r"C:\Users\xufc\Desktop\FaceGuard\my_face_model.yml"

target_size = (120, 120)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = []
labels = []
label = 0

# 先打印一下，确认目录对不对
print("人脸文件夹路径：", folder_path)
print("是否存在：", os.path.exists(folder_path))

# 加载图片
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join(folder_path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        detected_faces = face_cascade.detectMultiScale(img, 1.1, 5)
        for (x, y, w, h) in detected_faces:
            face_roi = img[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, target_size)
            face_roi = cv2.equalizeHist(face_roi)

            faces.append(face_roi)
            labels.append(label)

            # 数据增强：轻微旋转
            rows, cols = face_roi.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 3, 1)
            rotated = cv2.warpAffine(face_roi, M, (cols, rows))
            faces.append(rotated)
            labels.append(label)

print(f"✅ 加载并增强后，共 {len(faces)} 份人脸数据")

# 训练并保存
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save(model_path)
print("✅ 模型已保存到：", model_path)