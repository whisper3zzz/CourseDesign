# 智能人脸识别课程设计

这是一个基于 `PyQt5 + OpenCV + MindSpore` 的桌面端人脸识别项目，核心目标是完成从摄像头采集、样本管理、人脸检测到身份识别与结果展示的一体化流程。当前仓库重点保留了工程实现与模块结构，便于继续做课程设计答辩、功能扩展和算法替换。

## 项目定位

项目当前包含两条识别路线：

- 本地经典识别流程：使用 OpenCV 的经典方法完成训练与识别，适合快速验证功能闭环。
- 深度识别流程：使用 MindSpore 相关模型完成检测与识别能力扩展，适合继续做 CNN 路线的课程设计实现。

桌面端统一负责：

- 摄像头接入与实时画面展示
- 人脸检测框显示
- 本地样本采集与管理
- 本地训练与经典识别
- 深度识别结果接入
- 操作日志与状态反馈

## 技术栈

- 界面层：`PyQt5`
- 图像处理：`OpenCV`
- 轻量检测：`MediaPipe`
- 深度学习：`MindSpore`
- 可选服务端：`FastAPI`

## 整体架构

项目可以按下面几层理解：

1. 界面层
   负责主窗口布局、按钮交互、状态展示和结果渲染。
2. 服务层
   负责人脸检测、识别、样本注册、训练流程和远端服务对接。
3. 配置与状态层
   负责跨线程共享配置、识别模式和运行状态。
4. 工具层
   负责图像裁剪、训练数据准备以及 MindSpore 相关模型封装。

运行数据流如下：

```text
摄像头输入
  -> 图像采集
  -> 人脸检测
  -> 样本采集 / 实时识别
  -> 结果返回
  -> 界面展示与日志输出
```

## 目录结构

```text
CourseDesign/
├── main.py
├── view/
│   ├── main.py
│   ├── mainDisplay.py
│   └── mainDisplay.ui
├── service/
│   ├── vision.py
│   ├── ssh_tunnel.py
│   └── remote_sync.py
├── store/
│   └── config.py
├── utils/
│   ├── vision.py
│   └── mindface/
├── server/
│   ├── embedding_service.py
│   └── README.md
├── scripts/
│   ├── prepare_celeba_subset.py
│   ├── train_mobilefacenet_subset.py
│   ├── sync_local_samples_to_remote.py
│   └── start_with_ssh_tunnel.sh
├── dataset/
├── pretrained/
└── temp/
```

## 核心模块说明

### `main.py`

程序入口。负责创建 Qt 应用、初始化配置、构造视觉服务和主界面。

### `view/main.py`

主界面逻辑。负责：

- 窗口布局与视觉样式
- 摄像头状态切换
- 样本采集、删除、清空
- 训练按钮与识别按钮联动
- 结果区、日志区和状态提示刷新

### `service/vision.py`

项目核心服务模块。主要包括：

- 摄像头读取线程
- 人脸检测流程
- 经典识别模型训练与预测
- MindSpore 检测能力接入
- 样本注册与识别线程

### `store/config.py`

统一管理当前运行过程中的共享配置，包括：

- 是否开启检测
- 是否开启识别
- 当前检测方式
- 当前识别方式

### `utils/vision.py`

封装基础图像工具能力，包括：

- LBP 检测
- 人脸裁剪
- 训练数据整理

### `server/`

可选的远端识别服务模块。这个目录独立维护，适合在需要拆分桌面端和推理端时使用。更具体的服务结构见 [server/README.md](server/README.md)。

## 当前功能状态

桌面端已经具备以下完整流程：

1. 连接摄像头并显示实时画面
2. 开启检测并显示人脸区域
3. 输入姓名并采集当前人脸
4. 将样本保存到 `dataset/full/<name>/`
5. 使用本地样本训练经典识别模型
6. 在界面中显示识别结果和状态日志

在工程能力上，还补充了：

- 本地样本删除与整库清空
- 经典模型持久化与自动失效
- 启动时自动检查远端样本同步
- 可选 SSH 隧道启动支持

## 运行方式

推荐使用 `uv` 管理环境：

```bash
uv sync
uv run python main.py
```

如果需要 MindSpore 相关依赖：

```bash
uv sync --extra mindspore
```

如果已经有本地虚拟环境，也可以直接运行：

```bash
./.venv/bin/python main.py
```
