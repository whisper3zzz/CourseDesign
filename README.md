# 基于 OpenCV 的人脸识别系统

## 项目简介
这是一个用于课程设计的人脸识别桌面程序，基于 `PyQt5 + OpenCV` 实现，支持以下能力：

- 摄像头实时画面预览
- 人脸检测
- 本地样本采集
- 本地样本管理（删除当前样本 / 清空全部样本）
- 经典方法训练与识别
- 经典模型本地持久化与自动加载
- 经典识别陌生人阈值判断
- `MediaPipe` 轻量检测
- 可选 `MindSpore` 深度能力
- 远端 embedding 深度识别服务接入

课程背景为南京邮电大学智能科学与技术专业《智能系统课程设计》。

## 环境管理
项目当前使用 `uv` 管理 Python 环境。

推荐命令：

```bash
uv sync
uv run python main.py
```

当前仓库默认使用 Python `3.10`，版本固定在 `.python-version`。

默认依赖已经覆盖当前仓库的基础运行能力，包括：

- `PyQt5`
- `opencv-contrib-python`
- `mediapipe`
- `numpy`
- `pillow`
- `requests`

如需启用 `MindSpore` 相关能力，可额外安装：

```bash
uv sync --extra mindspore
```

如果当前平台不支持 `MindSpore`，程序仍可正常使用经典检测、经典识别和 `MediaPipe` 轻量检测。

## 深度识别服务接入
当前仓库里的“深度识别”已经切成远端 embedding 服务模式。

桌面端默认会请求：

```bash
http://127.0.0.1:18000
```

这个地址不是公网服务地址，而是给本机 SSH 隧道预留的默认入口。这样做的原因是很多课程设计服务器只开放了 SSH 端口，没有直接开放 `8000`。

推荐连接方式：

```bash
ssh -o StrictHostKeyChecking=no -L 18000:127.0.0.1:8000 <user>@<server-host> -p <ssh-port> -N
```

隧道建立后，桌面程序直接启动即可：

```bash
uv run python main.py
```

如果你已经把服务端 `8000` 端口直接暴露到外网，也可以显式覆盖：

```bash
FACE_SERVICE_BASE_URL=http://<server-host>:8000 uv run python main.py
```

查看服务状态：

```bash
curl http://127.0.0.1:18000/health
```

如果能返回 `status/device/model/threshold` 这些字段，说明桌面端可以连通远端深度识别服务。

## CNN 识别落地路线
当前仓库已经从 `PubFig` 路线切到更适合真正落地的 `CelebA` 路线。

原因很直接：

- `PubFig` 官方只提供图片 URL 清单，今天大量链接已经失效
- `CelebA` 官方长期可获取，且自带对齐后的人脸图
- 更适合在本仓库里裁出一个小子集，做本地 `MobileFaceNet` 训练

推荐流程：

1. 从官方页面准备 `CelebA` 对齐图和标注文件到 `data/celeba/raw/`
2. 用仓库脚本生成一个适合 CPU 的小规模身份子集
3. 用本地 `MindSpore + MobileFaceNet` 训练分类头或微调 backbone
4. 后续把训练出的 backbone 接回桌面程序做 embedding 比对

官方页面：

`https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html`

子集准备命令：

```bash
uv run python scripts/prepare_celeba_subset.py \
  --raw-root data/celeba/raw \
  --output-root data/celeba/subsets/course_v1 \
  --max-identities 50 \
  --train-per-identity 20 \
  --val-per-identity 5 \
  --test-per-identity 5
```

本地训练命令：

```bash
uv sync --extra mindspore
uv run python scripts/train_mobilefacenet_subset.py \
  --train-dir data/celeba/subsets/course_v1/train \
  --val-dir data/celeba/subsets/course_v1/val \
  --output-dir output/mobilefacenet_course_v1
```

补充说明：

- 当前这台 macOS 环境已经实测安装好了 `MindSpore 2.8.0`
- 但 `Conv2d` 在 CPU 上会触发底层 `Xbyak::Error`
- 所以仓库里的训练脚本会先做卷积自检，失败时直接提示环境不兼容
- 也就是说，`CelebA` 数据准备这一步可以本机完成，真正的 `MindSpore CNN` 训练更适合放到 `Linux/GPU` 环境

更详细的数据目录说明见 `data/celeba/README.md`。

## 启动方式
首次进入仓库时建议先同步依赖：

```bash
uv sync
```

启动桌面程序：

```bash
uv run python main.py
```

如果你已经有 `.venv`，也可以直接使用：

```bash
./.venv/bin/python main.py
```

## 推荐使用流程
当前版本最顺畅的使用顺序如下。

1. 如果要使用“深度识别”，先连好 SSH 隧道或设置好 `FACE_SERVICE_BASE_URL`。
2. 点击“连接摄像头”。
3. 建议先开启“人脸检测”，检测方式优先选择“轻量检测”。
4. 在左侧输入姓名，点击“采集当前人脸”。
5. 如果你走“经典识别”，采集成功后点击“用本地样本训练”。
6. 如果你走“深度识别”，不需要本机训练，直接开启“人脸识别”即可。
7. 如果识别方式选择“经典识别”，必须先完成训练。

说明：

- “采集当前人脸”会优先使用当前锁定的人脸；如果当前未锁定，会自动尝试多种检测方式寻找人脸。
- 采集成功后，本地样本会保存到 `dataset/full/<姓名>/`。
- 点击“删除当前样本”会删除当前姓名对应的本地样本目录。
- 点击“清空全部样本”会清空 `dataset/full/` 下全部本地样本。
- 每次训练完成后，经典模型会自动保存到 `pretrained/classic_lbph.yml` 和 `pretrained/classic_lbph.json`。
- 程序下次启动时，如果本地样本没有变化，会自动加载上一次保存的经典模型。
- 如果本地样本发生变化，旧经典模型会自动失效，需要重新训练。
- 深度识别和远端注册走同一个 embedding 服务；服务不可用时，本地采集样本仍会保留，不影响经典训练。

## 数据目录
当前仓库和运行流程中，比较重要的目录如下：

- `dataset/full/`
  - 本地采集样本目录
  - 每个人一个子目录，例如 `dataset/full/张三/`
- `pretrained/`
  - OpenCV 经典检测器等预训练文件
  - 经典识别模型持久化文件 `classic_lbph.yml` / `classic_lbph.json`
- `service/`
  - 摄像头、采集、训练、识别等服务逻辑
- `view/`
  - PyQt 界面与布局逻辑
- `store/`
  - 运行状态与共享配置
- `utils/`
  - 人脸裁剪、经典检测、MediaPipe、MindSpore 相关工具

## 当前交互逻辑说明
为了减少误操作，界面上做了以下约束：

- 摄像头未连接时，采集、检测、识别相关操作会自动禁用，但仍可离线查看或删除本地样本。
- 经典识别模型未训练完成时，经典识别不会允许开启。
- 本地样本变化后，旧经典模型会自动清除，避免误用过期模型。
- 关闭人脸检测时，会同步关闭人脸识别。
- 训练进行中时，训练按钮会显示“训练中...”。
- 左侧步骤提示会根据当前状态自动更新。
- 经典识别命中距离超过阈值时，会显示“未知”，不会强行匹配到错误身份。

## 常见问题
### 1. 采集成功了，但点击本地训练提示没有可用样本
常见原因不是“没保存成功”，而是训练阶段在样本图里二次做人脸检测失败。

当前版本已经修正为：

- 如果训练阶段检测不到人脸，就直接把采集到的整张样本图作为训练输入

如果你是在旧版本基础上运行，需要完全退出程序后重新启动，再重新点击“用本地样本训练”。

### 2. 摄像头有画面，但采集时提示没有锁定到人脸
优先检查以下几点：

- 先开启“人脸检测”
- 检测方式优先选“轻量检测”
- 保证脸在画面中央，距离不要太远
- 尽量避免逆光和过暗环境

### 3. 经典识别为什么不能直接开
经典识别依赖本地训练好的模型。没有训练完成时，程序会阻止开启经典识别，避免直接触发 OpenCV 的未训练模型异常。

### 4. 删除 `dataset/full` 里的文件后，还要做什么
现在程序会在下次启动或下一次样本变更时自动判断经典模型是否过期。

如果你手动删除了 `dataset/full` 里的样本：

- 旧经典模型不会继续被当成有效模型使用
- 需要重新采集并重新训练

更推荐直接用界面里的“删除当前样本”或“清空全部样本”，这样界面状态、日志和模型失效逻辑都会一起更新。

### 5. 终端里出现 `MediaPipe` 或字体警告
这类信息通常是底层库的提示，不一定代表程序错误。常见如：

- `Feedback manager requires a model with a single signature inference`
- `Replace uses of missing font family`

通常不会影响摄像头画面、采集和训练主流程。

### 6. 为什么深度识别连不上
先按顺序检查：

- 远端服务是否存活：`curl http://127.0.0.1:18000/health`
- SSH 隧道是否已建立：本机是否监听 `127.0.0.1:18000`
- 如果没走隧道，`FACE_SERVICE_BASE_URL` 是否已经改成真实服务地址
- 如果只打算用“经典识别”，可以完全不启用远端服务

## 程序架构
整体可以按以下层次理解：

- `View`
  - 界面展示
  - 用户交互
  - 日志输出
- `Store`
  - 全局配置与运行状态
- `Service`
  - 摄像头读取
  - 人脸检测
  - 经典训练与识别
  - 远端 embedding 注册与识别请求
- `Utils`
  - 图像裁剪
  - OpenCV 检测工具
  - `MediaPipe` 封装
  - `MindSpore` 与深度识别相关工具

## 如何阅读代码
推荐从 `main.py` 开始进入：

```python
store = ConfigStore()
classicPreditor = ClassicFaceRecognizer()
camera = VisionService(store)
window = MainPage(camera.all_queues, classicPreditor, camera, store)
```

阅读顺序建议：

1. `main.py`
2. `view/main.py`
3. `service/vision.py`
4. `utils/vision.py`
5. `store/config.py`

如果要改功能逻辑，优先看 `service/vision.py`；如果要改界面和交互，优先看 `view/main.py`。

## 说明
本项目现在已经内置了一套可单独部署的远端 embedding 服务，代码位于 `server/`。即使远端服务不可用，本地采集、训练和经典识别流程仍然可以独立工作。
