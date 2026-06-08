import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import time
import tkinter as tk
from tkinter import ttk
import threading

# ======================【配置区】======================
FACE_MODEL_PATH = r"C:\Users\xufc\Desktop\FaceGuard\my_face_model.yml"
ALERT_MINUTES = 30
AWAY_SHORT_SECONDS = 5      # 短暂离开：5秒 → 暂停计时
AWAY_LONG_SECONDS = 300     # 长时间离开：5分钟 → 视为已休息，重置
CHECK_INTERVAL = 1
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
# =====================================================

start_time = time.time()
is_away = False
away_start_time = None
resting = False
rest_start_time = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(FACE_MODEL_PATH)

face_history = deque(maxlen=10)

try:
    font = ImageFont.truetype(FONT_PATH, 24)
    small_font = ImageFont.truetype(FONT_PATH, 20)
except:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

def put_chinese(img, text, pos, font_obj, color=(0,255,0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font_obj, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def show_rest_reminder():
    global start_time
    root = tk.Tk()
    root.title("FaceGuard 休息提醒")
    root.geometry("350x180")
    root.attributes('-topmost', True)

    def rest_now():
        root.destroy()
        start_resting()

    def keep_working():
        root.destroy()
        start_time = time.time()

    tk.Label(root, text="🧍‍♂️ 你已连续工作 30 分钟", font=("SimHei", 14)).pack(pady=10)
    tk.Label(root, text="请休息一下，保护眼睛和腰椎", font=("SimHei", 12)).pack()
    ttk.Button(root, text="现在去休息", command=rest_now).pack(pady=5, fill=tk.X)
    ttk.Button(root, text="继续工作", command=keep_working).pack(pady=3, fill=tk.X)
    root.mainloop()

def start_resting():
    global resting, rest_start_time
    resting = True
    rest_start_time = time.time()
    root = tk.Tk()
    root.title("休息中")
    root.geometry("300x150")
    root.attributes('-topmost', True)

    time_label = tk.Label(root, text="", font=("SimHei", 14))
    time_label.pack(pady=20)

    def end_rest():
        global resting, start_time
        resting = False
        start_time = time.time()
        root.destroy()

    ttk.Button(root, text="✅ 结束休息，重新计时", command=end_rest).pack(fill=tk.X, padx=20)

    def update_time():
        if not resting:
            return
        elapsed = int(time.time() - rest_start_time)
        time_label.config(text=f"已休息：{elapsed} 秒")
        root.after(500, update_time)

    update_time()
    root.mainloop()

def check_time():
    global start_time, is_away, away_start_time
    while True:
        if resting:
            time.sleep(0.5)
            continue

        current_is_me = sum(face_history) > 5

        if current_is_me:
            # 重新出现
            if is_away:
                away_duration = time.time() - away_start_time
                if away_duration >= AWAY_LONG_SECONDS:
                    # 离开超过5分钟 → 视为已经休息过 → 重置
                    start_time = time.time()
                    print("🛑 离开超过5分钟，已自动重置工作计时")
                is_away = False
                away_start_time = None

            elapsed = time.time() - start_time
            if elapsed >= ALERT_MINUTES * 60:
                threading.Thread(target=show_rest_reminder, daemon=True).start()
                start_time = time.time()

        else:
            if not is_away:
                if away_start_time is None:
                    away_start_time = time.time()
                else:
                    if time.time() - away_start_time > AWAY_SHORT_SECONDS:
                        is_away = True

        time.sleep(CHECK_INTERVAL)

def camera_loop():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("FaceGuard - Smart Rest Reminder", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        is_me_current = False
        confidence = 100.0
        x, y, w, h = 0,0,0,0

        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (120,120))
            roi = cv2.equalizeHist(roi)
            label, confidence = recognizer.predict(roi)
            is_me_current = confidence < 85

            color = (0,255,0) if is_me_current else (0,0,255)
            text = "是你本人" if is_me_current else "不是你"
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            frame = put_chinese(frame, f"{text} {confidence:.1f}", (x, y-30), font, color)

        face_history.append(is_me_current)

        status = "工作中" if not is_away else "已离开"
        elapsed = int(time.time() - start_time)
        status_text = f"状态：{status}  计时：{elapsed}s"
        frame = put_chinese(frame, status_text, (10, 40), small_font, (255,255,255))

        cv2.imshow("FaceGuard - Smart Rest Reminder", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("✅ FaceGuard 已启动")
    print("✅ 短暂离开5秒 → 暂停计时")
    print("✅ 离开超过5分钟 → 自动重置（视为已休息）")
    print("✅ 连续工作30分钟 → 提醒休息")
    print("按 q 退出\n")

    threading.Thread(target=check_time, daemon=True).start()
    camera_loop()