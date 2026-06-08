import sys
import os
import cv2
import time
import logging
from collections import deque
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# ====================== 全局配置 ======================
DEFAULT_DIR = r"C:\Users\xufc\Desktop\FaceGuard"
LOG_FILE = os.path.join(DEFAULT_DIR, "faceguard.log")

# ================ 日志系统 ================
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def fmt_sec(sec):
    """将秒数格式化为 Xm Xs 或 Xs"""
    sec = max(0, int(sec))
    m = sec // 60
    s = sec % 60
    if m > 0:
        return f"{m}分{s:02d}秒"
    return f"{s}秒"


# ================ 核心工作线程 ================
class Worker(QThread):
    # 状态信号：状态文字, 已工作秒, 已离开秒, 距提醒秒
    status_signal = pyqtSignal(str, int, int, int)
    rest_signal = pyqtSignal(int)    # 触发休息提醒，携带已工作秒数
    log_signal = pyqtSignal(str)     # 实时日志信号

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = True
        self.resting = False         # 休息中时暂停状态信号

        self.start_time = time.time()
        self.last_detected_me_time = time.time()
        self.is_away = False
        self.away_start_time = None  # 记录本次离开的开始时间

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(config["model_path"])

        self.face_history = deque(maxlen=15)

    def emit_log(self, msg):
        log.info(msg)
        self.log_signal.emit(msg)

    def run(self):
        self.emit_log("监测已启动")
        self.emit_log(
            f"配置 → 提醒间隔: {fmt_sec(self.config['alert'])}  |  "
            f"短暂离开: {self.config['away_short']}秒  |  "
            f"长离开重置: {fmt_sec(self.config['away_long'])}"
        )

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.emit_log("[错误] 无法打开摄像头，请检查设备")
            return

        self.emit_log("摄像头就绪，开始计时")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            has_me = False

            if len(faces) > 0:
                x, y, w, h = faces[0]
                roi = cv2.resize(gray[y:y+h, x:x+w], (120, 120))
                roi = cv2.equalizeHist(roi)
                _, conf = self.recognizer.predict(roi)
                has_me = conf < 85

            if has_me:
                self.last_detected_me_time = time.time()

            self.face_history.append(has_me)
            now = time.time()
            lost = now - self.last_detected_me_time
            present = lost < self.config["face_lost"]

            # 休息中时不更新状态信号，避免干扰休息计时界面
            if self.resting:
                time.sleep(0.2)
                continue

            # ========== 状态更新 ==========
            if present:
                if self.is_away:
                    actual_away = now - self.away_start_time if self.away_start_time else 0
                    if actual_away >= self.config["away_long"]:
                        self.emit_log(
                            f"回到座位 ← 离开了 {fmt_sec(actual_away)}，"
                            f"超过长离开阈值（{fmt_sec(self.config['away_long'])}），重置计时"
                        )
                        self.start_time = now
                    else:
                        self.emit_log(
                            f"回到座位 ← 离开了 {fmt_sec(actual_away)}，继续计时"
                        )
                    self.is_away = False
                    self.away_start_time = None

                work_sec = int(now - self.start_time)
                next_alert = max(0, int(self.config["alert"]) - work_sec)
                self.status_signal.emit("工作中", work_sec, 0, next_alert)

                if work_sec >= self.config["alert"]:
                    self.emit_log(f"连续工作 {fmt_sec(work_sec)}，触发休息提醒")
                    self.resting = True
                    self.rest_signal.emit(work_sec)
                    self.start_time = now

            else:
                if not self.is_away and lost >= self.config["away_short"]:
                    work_sec = int(now - self.start_time)
                    self.emit_log(
                        f"检测到离开（>{self.config['away_short']}秒），暂停计时"
                        f"（本次已工作 {fmt_sec(work_sec)}）"
                    )
                    self.is_away = True
                    self.away_start_time = now

                away_sec = int(now - self.away_start_time) if self.away_start_time else int(lost)
                self.status_signal.emit("已离开", 0, away_sec, 0)

            # ========== 摄像头窗口 ==========
            if self.config["show_window"]:
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    if has_me:
                        color = (0, 255, 0)    # 绿色：识别为本人
                        label_text = "Me"
                    else:
                        color = (0, 165, 255)  # 橙色：检测到人脸但不确定
                        label_text = "Unknown"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label_text, (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("FaceGuard Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.05)

        cap.release()
        cv2.destroyAllWindows()
        self.emit_log("监测已停止")

    @pyqtSlot()
    def resume_after_rest(self):
        """用户点击'继续工作'后调用，重置计时并恢复状态信号"""
        now = time.time()
        self.start_time = now
        self.last_detected_me_time = now
        self.is_away = False
        self.away_start_time = None
        self.resting = False
        self.emit_log("休息结束，计时已重置，继续工作")


# ================ 休息提醒窗口 ================
class RestWindow(QWidget):
    """提示用户去休息，点击按钮后打开休息计时窗口"""
    confirmed = pyqtSignal(int)   # 用户确认去休息，携带已工作秒

    def __init__(self):
        super().__init__()
        self.setWindowTitle("休息提醒")
        self.setFixedSize(420, 210)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self._work_sec = 0

        layout = QVBoxLayout()
        layout.setSpacing(12)

        self.main_label = QLabel("工作时间到，请休息一下！")
        self.main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_label.setStyleSheet("font-size:20px; color:#c0392b; font-weight:bold;")

        self.sub_label = QLabel("")
        self.sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub_label.setStyleSheet("font-size:13px; color:#7f8c8d;")

        btn = QPushButton("我知道了，去休息")
        btn.clicked.connect(self._on_confirm)
        btn.setStyleSheet(
            "font-size:14px; padding:10px; "
            "background:#e74c3c; color:white; border-radius:6px;"
        )

        layout.addWidget(self.main_label)
        layout.addWidget(self.sub_label)
        layout.addWidget(btn)
        self.setLayout(layout)

    def show_with_info(self, work_sec):
        self._work_sec = work_sec
        self.sub_label.setText(f"本次连续工作时长：{fmt_sec(work_sec)}")
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_confirm(self):
        self.hide()
        self.confirmed.emit(self._work_sec)


# ================ 休息计时窗口 ================
class RestingWindow(QWidget):
    """显示已休息时长，提供'继续工作'按钮"""
    resume_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("休息中")
        self.setFixedSize(380, 220)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self._rest_start = 0
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

        layout = QVBoxLayout()
        layout.setSpacing(14)

        title = QLabel("休息中...")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:18px; color:#2980b9; font-weight:bold;")

        self.work_hint = QLabel("")
        self.work_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.work_hint.setStyleSheet("font-size:12px; color:#7f8c8d;")

        self.rest_label = QLabel("已休息：0 秒")
        self.rest_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rest_label.setStyleSheet("font-size:22px; color:#27ae60; font-weight:bold;")

        btn = QPushButton("继续工作")
        btn.clicked.connect(self._on_resume)
        btn.setStyleSheet(
            "font-size:14px; padding:10px; "
            "background:#27ae60; color:white; border-radius:6px;"
        )

        layout.addWidget(title)
        layout.addWidget(self.work_hint)
        layout.addWidget(self.rest_label)
        layout.addWidget(btn)
        self.setLayout(layout)

    def start_rest(self, work_sec):
        self._rest_start = time.time()
        self.work_hint.setText(f"刚才连续工作了 {fmt_sec(work_sec)}，好好放松一下")
        self.rest_label.setText("已休息：0 秒")
        self._timer.start()
        self.show()
        self.raise_()
        self.activateWindow()

    def _tick(self):
        elapsed = int(time.time() - self._rest_start)
        self.rest_label.setText(f"已休息：{fmt_sec(elapsed)}")

    def _on_resume(self):
        self._timer.stop()
        self.hide()
        self.resume_clicked.emit()


# ================ 实时状态窗口（半透明浮窗，可拖动） ================
class StatusWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceGuard 实时状态")
        # 无边框 + 始终置顶 + 不在任务栏显示
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        # 背景透明（结合 paintEvent 绘制半透明圆角背景）
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(240, 140)

        self._drag_pos = None   # 记录鼠标拖动起点

        # 内容布局（加内边距让圆角背景留出空间）
        layout = QVBoxLayout()
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(5)

        self.status_label = QLabel("状态：等待开始")
        self.work_label   = QLabel("已工作：--")
        self.away_label   = QLabel("已离开：--")
        self.next_label   = QLabel("距提醒：--")

        _base = "font-size:13px; color:white;"
        for lbl in [self.status_label, self.work_label, self.away_label, self.next_label]:
            lbl.setStyleSheet(_base)
            layout.addWidget(lbl)

        self.setLayout(layout)

    def paintEvent(self, _):
        """绘制半透明圆角黑色背景"""
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(20, 20, 20, 185))   # 黑色，约73%不透明
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 12, 12)

    # ---- 鼠标拖动 ----
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, _):
        self._drag_pos = None

    def update_info(self, status, work_sec, away_sec, next_alert_sec):
        self.status_label.setText(f"状态：{status}")
        if status == "工作中":
            self.status_label.setStyleSheet("font-size:13px; color:#2ecc71; font-weight:bold;")
            self.work_label.setText(f"已工作：{fmt_sec(work_sec)}")
            self.away_label.setText("已离开：--")
            self.next_label.setText(f"距提醒：{fmt_sec(next_alert_sec)}")
        elif status == "休息中":
            self.status_label.setStyleSheet("font-size:13px; color:#3498db; font-weight:bold;")
            self.work_label.setText("已工作：（休息中）")
            self.away_label.setText("已离开：--")
            self.next_label.setText("距提醒：（休息中）")
        else:
            self.status_label.setStyleSheet("font-size:13px; color:#f39c12; font-weight:bold;")
            self.work_label.setText("已工作：（已暂停）")
            self.away_label.setText(f"已离开：{fmt_sec(away_sec)}")
            self.next_label.setText("距提醒：（已暂停）")
        # 其余标签统一白色
        for lbl in [self.work_label, self.away_label, self.next_label]:
            lbl.setStyleSheet("font-size:13px; color:white;")


# ================ 主设置界面 ================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceGuard 智能久坐提醒")
        self.setFixedSize(540, 580)
        self.worker = None
        self.status_win = StatusWindow()
        self.rest_win = RestWindow()
        self.resting_win = RestingWindow()

        # 信号串联
        self.rest_win.confirmed.connect(self._on_rest_confirmed)
        self.resting_win.resume_clicked.connect(self._on_resume_work)

        self._init_tray()
        self.init_ui()

    def _init_tray(self):
        """创建系统托盘图标"""
        # 用程序内置图标生成一个简单的托盘图标（绿色圆点）
        pix = QPixmap(16, 16)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#27ae60"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, 14, 14)
        painter.end()
        tray_icon = QIcon(pix)

        self.tray = QSystemTrayIcon(tray_icon, self)
        self.tray.setToolTip("FaceGuard 智能久坐提醒")

        menu = QMenu()
        action_show = menu.addAction("显示主窗口")
        action_show.triggered.connect(self._show_from_tray)
        menu.addSeparator()
        action_quit = menu.addAction("退出")
        action_quit.triggered.connect(self._quit)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

    def init_ui(self):
        w = QWidget()
        self.setCentralWidget(w)
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # 1. 目录选择
        layout.addWidget(QLabel("工作目录"))
        self.dir_edit = QLineEdit(DEFAULT_DIR)
        self.dir_btn = QPushButton("选择目录")
        self.dir_btn.clicked.connect(self.choose_dir)
        h = QHBoxLayout()
        h.addWidget(self.dir_edit)
        h.addWidget(self.dir_btn)
        layout.addLayout(h)

        # 2. 摄像头选项
        self.show_camera = QCheckBox("显示摄像头实时画面")
        layout.addWidget(self.show_camera)

        # 3. 参数设置
        layout.addWidget(QLabel("参数设置（单位：秒）"))
        form = QFormLayout()
        self.alert      = QLineEdit("120")
        self.away_short = QLineEdit("5")
        self.away_long  = QLineEdit("30")
        self.face_lost  = QLineEdit("1.5")
        form.addRow("连续工作提醒（秒）：", self.alert)
        form.addRow("短暂离开判定（秒）：", self.away_short)
        form.addRow("长离开重置计时（秒）：", self.away_long)
        form.addRow("人脸丢失判定（秒）：", self.face_lost)
        layout.addLayout(form)

        # 4. 按钮行
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  开始运行")
        self.start_btn.clicked.connect(self.start)
        self.start_btn.setStyleSheet(
            "font-size:14px; padding:10px; "
            "background:#27ae60; color:white; border-radius:6px;"
        )
        self.stop_btn = QPushButton("■  停止")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "font-size:14px; padding:10px; "
            "background:#c0392b; color:white; border-radius:6px;"
        )
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        # 5. 状态提示
        self.tip = QLabel("状态：未启动")
        layout.addWidget(self.tip)

        # 6. 实时日志面板
        layout.addWidget(QLabel("实时日志："))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(160)
        self.log_box.setStyleSheet(
            "font-size:12px; background:#1e1e1e; color:#d4d4d4; "
            "font-family:Consolas,monospace; border-radius:4px;"
        )
        layout.addWidget(self.log_box)

    def choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择目录", DEFAULT_DIR)
        if d:
            self.dir_edit.setText(d)

    def append_log(self, msg):
        now_str = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{now_str}]  {msg}")
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    def start(self):
        try:
            root = self.dir_edit.text()
            model_path = os.path.join(root, "my_face_model.yml")
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "错误", "未找到模型文件：my_face_model.yml")
                return

            config = {
                "model_path": model_path,
                "alert":      float(self.alert.text()),
                "away_short": float(self.away_short.text()),
                "away_long":  float(self.away_long.text()),
                "face_lost":  float(self.face_lost.text()),
                "show_window": self.show_camera.isChecked(),
            }

            self.worker = Worker(config)
            self.worker.status_signal.connect(self.status_win.update_info)
            self.worker.rest_signal.connect(self.rest_win.show_with_info)
            self.worker.log_signal.connect(self.append_log)
            self.worker.start()
            self.status_win.show()
            self.tip.setText("状态：运行中 ✓")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

        except ValueError:
            QMessageBox.warning(self, "错误", "参数输入有误，请检查是否为数字")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _on_rest_confirmed(self, work_sec):
        """用户在提醒窗点击'去休息'"""
        self.status_win.update_info("休息中", 0, 0, 0)
        self.append_log(f"用户开始休息（之前工作了 {fmt_sec(work_sec)}）")
        self.resting_win.start_rest(work_sec)

    def _on_resume_work(self):
        """用户在休息计时窗点击'继续工作'"""
        rest_sec = int(time.time() - self.resting_win._rest_start)
        self.append_log(f"休息结束，共休息了 {fmt_sec(rest_sec)}，继续计时")
        if self.worker:
            self.worker.resume_after_rest()

    def _tray_activated(self, reason):
        """双击托盘图标恢复窗口"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._show_from_tray()

    def _show_from_tray(self):
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _quit(self):
        self.stop()
        QApplication.quit()

    def changeEvent(self, event):
        """最小化时隐藏到托盘"""
        if event.type() == QEvent.Type.WindowStateChange:
            if self.isMinimized():
                event.ignore()
                self.hide()
                self.tray.showMessage(
                    "FaceGuard",
                    "已最小化到托盘，双击图标可恢复",
                    QSystemTrayIcon.MessageIcon.Information,
                    2000
                )
                return
        super().changeEvent(event)

    def stop(self):
        if self.worker:
            self.worker.running = False
            self.worker.wait(2000)
            self.worker = None
        self.status_win.hide()
        self.tip.setText("状态：已停止")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log("用户手动停止监测")

    def closeEvent(self, event):
        self.stop()
        self.tray.hide()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
