# FaceGuard 智能久坐提醒

基于人脸识别的久坐提醒工具。通过摄像头实时检测你是否在座位前，连续工作达到设定时长后自动弹窗提醒休息，并记录每次的工作和休息时长。

---

## 环境依赖

```bash
pip install opencv-python opencv-contrib-python pillow PyQt6
```

> `opencv-contrib-python` 包含 LBPH 人脸识别器，缺少此包训练/识别会报错。

---

## 使用流程

### 第一步：采集人脸数据

运行 `test_camera.py`，摄像头会打开实时画面：

```bash
python test_camera.py
```

- 画面中检测到人脸时会出现绿色框
- **按 `S` 键** 保存当前帧人脸图片到 `my_face/` 目录
- **按 `Q` 键** 退出采集

**建议采集数量：50～100 张**，多角度（正面、略偏左右、略仰俯）效果更好。  
重复运行不会覆盖已有图片，自动从上次最大编号续接。

---

### 第二步：训练模型

运行 `train_face.py`，读取 `my_face/` 中的图片训练 LBPH 识别模型：

```bash
python train_face.py
```

训练完成后会在项目目录生成 `my_face_model.yml`（约 10～20 MB）。  
每次新增采集数据后都需要重新训练。

---

### 第三步：启动主程序

运行主界面：

```bash
python test_recognize_me.py
```

---

## 主界面参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 工作目录 | 程序所在目录 | 指定 `my_face_model.yml` 所在位置 |
| 连续工作提醒（秒） | 120 | 连续工作多少秒后弹出休息提醒 |
| 短暂离开判定（秒） | 5 | 离开超过此秒数才暂停工作计时 |
| 长离开重置计时（秒） | 30 | 离开超过此秒数回来后重置工作计时 |
| 人脸丢失判定（秒） | 1.5 | 人脸消失超过此时间才判定为离开 |
| 显示摄像头实时画面 | 否 | 勾选后弹出 OpenCV 摄像头调试窗口 |

点击 **▶ 开始运行** 启动监测，点击 **■ 停止** 结束。

---

## 窗口说明

### 主窗口
- 最小化后自动隐藏到右下角**系统托盘**（绿色圆点图标）
- 托盘图标**双击**恢复主窗口，**右键**可选择"显示主窗口"或"退出"

### 实时状态浮窗
- 半透明黑色圆角悬浮窗，**始终置顶**，不显示在任务栏
- **鼠标拖动**可移动到屏幕任意位置（建议放在屏幕中上方）
- 显示当前状态（工作中 / 已离开 / 休息中）、已工作时长、已离开时长、距下次提醒时长

### 休息提醒流程
1. 达到提醒时长 → 弹出**休息提醒窗口**
2. 点击"我知道了，去休息" → 弹出**休息计时窗口**，实时显示已休息时长
3. 休息完毕点击"继续工作" → 工作计时重置，重新开始监测

---

## 文件结构

```
FaceGuard/
├── test_recognize_me.py   # 主程序（运行这个）
├── test_camera.py         # 人脸数据采集
├── train_face.py          # 模型训练
├── my_face_model.yml      # 训练好的模型（训练后生成）
├── my_face/               # 采集的人脸图片（采集后生成）
│   └── face_1.jpg ...
└── faceguard.log          # 运行日志（运行后生成）
```

---

## 打包成 exe

使用虚拟环境中的 PyInstaller 打包，打包前确保虚拟环境已安装所有依赖。

### 安装依赖（首次）

```bash
I:\plane_game\games\Scripts\pip.exe install opencv-python opencv-contrib-python PyQt6 pillow pyinstaller
```

### 执行打包

在项目根目录执行：

```bash
"I:\plane_game\games\Scripts\pyinstaller.exe" --noconfirm --onedir --windowed --name "FaceGuard" --add-data "I:\plane_game\games\lib\site-packages\cv2\data;cv2\data" test_recognize_me.py
```

打包完成后输出目录为 `dist\FaceGuard\`。

### 打包后的准备

将 `my_face_model.yml` 复制到 `dist\FaceGuard\` 目录，与 `FaceGuard.exe` 放在一起：

```
dist\FaceGuard\
├── FaceGuard.exe        ← 双击运行
├── my_face_model.yml    ← 必须复制过来
└── _internal\           ← 依赖库，勿删
```

### 注意事项

- 每次修改代码后需重新打包，并重新复制 `my_face_model.yml`
- 重新打包会自动覆盖 `dist\FaceGuard\`，无需手动删除
- `my_face/` 和 `my_face_model.yml` 已加入 `.gitignore`，不会上传到 git

---

## 常见问题

**Q：启动后一直显示"已离开"，无法识别**  
A：确认 `my_face_model.yml` 存在；采集样本不足时识别率低，建议补采 50 张以上后重新训练。

**Q：识别到了但把别人也当成我**  
A：降低置信度阈值（代码中 `conf < 85`，适当改小，如 `conf < 70`）。

**Q：提示"无法打开摄像头"**  
A：检查摄像头是否被其他程序占用，或尝试将 `cv2.VideoCapture(0)` 中的 `0` 改为 `1`。
