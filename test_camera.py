import cv2
import os

# 你的保存目录
SAVE_DIR = r"C:\Users\xufc\Desktop\FaceGuard\my_face"

# 自动创建文件夹
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ==============================================
# 自动计算当前已经有多少张图片，下次从最后一张+1开始
# ==============================================
count = 0
for filename in os.listdir(SAVE_DIR):
    if filename.startswith("face_") and filename.endswith(".jpg"):
        # 提取数字
        try:
            num = int(filename.replace("face_", "").replace(".jpg", ""))
            if num > count:
                count = num
        except:
            pass

print(f"✅ 检测到已有 {count} 张图片，下次将从第 {count + 1} 张开始")

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("按 s 键保存照片，按 q 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    face_img = None

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_img = frame[y:y + h, x:x + w]

    cv2.imshow("采集人脸", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if face_img is not None:
            count += 1
            save_path = os.path.join(SAVE_DIR, f"face_{count}.jpg")
            cv2.imwrite(save_path, face_img)
            print(f"✅ 已保存：face_{count}.jpg")
        else:
            print("❌ 未检测到人脸，无法保存")

cap.release()
cv2.destroyAllWindows()