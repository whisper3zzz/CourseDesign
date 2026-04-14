# CourseDesign 架构说明

这个仓库当前是一套以桌面端为主、远端推理服务可选拆分的人脸识别系统。根目录 README 只描述代码结构、模块边界和运行链路，不展开课程背景、操作流程或部署细节。

## 架构概览

系统按职责分为六层：

1. 桌面应用入口层
   `main.py` 负责启动 Qt 应用、初始化共享配置、创建视觉服务，并在启动阶段接入 SSH 隧道自举逻辑。
2. 界面编排层
   `view/main.py` 负责主窗口布局、按钮交互、状态展示、日志输出，以及应用启动后的远端样本同步触发。
3. 本地视觉服务层
   `service/vision.py` 负责摄像头读取、检测链路、注册链路、识别链路、本地经典模型训练，以及和远端识别服务之间的数据交换。
4. 共享状态与调度层
   `store/config.py` 维护跨线程共享的开关状态和识别模式，并暴露识别队列、注册队列等运行时通道。
5. 算法与资源层
   `utils/`、`pretrained/`、`dataset/`、`temp/` 提供本地检测/裁剪工具、MindSpore 相关模型代码、经典检测器资源、样本数据和同步中间产物。
6. 可选远端服务层
   `server/` 提供可独立部署的 HTTP 推理服务，负责远端身份库维护、embedding 识别、CNN 分类器识别以及服务端运行时状态管理。

## 部署形态

仓库支持两种边界清晰的运行形态：

### 1. 纯桌面端闭环

- `main.py`
- `view/main.py`
- `service/vision.py`
- `store/config.py`
- `utils/vision.py`
- `dataset/full/`

这一路径由桌面端独立完成摄像头采集、经典检测、样本落盘、本地经典模型训练和本地经典识别。

### 2. 桌面端 + 远端服务

- 桌面端仍负责采集、UI 和流程调度
- `service/vision.py` 的识别/注册线程通过 HTTP 调用 `server/embedding_service.py`
- `service/remote_sync.py` 负责把 `dataset/full/` 与远端身份库做增量重建和删除同步
- `service/ssh_tunnel.py` 在本地服务地址场景下可自动建立 SSH 端口转发

这种结构把 GUI、摄像头处理、本地样本管理和远端推理解耦开，便于分别演进。

## 运行链路

### 桌面端主链路

```text
main.py
  -> ConfigStore
  -> VisionService
  -> MainPage
  -> 摄像头读取线程 / 注册线程 / 识别线程
  -> UI 展示与日志反馈
```

### 数据链路

```text
摄像头帧
  -> ReadCameraThread
  -> 人脸检测
  -> 裁剪人脸区域
  -> 分流
     -> 本地注册: dataset/full/<identity>/
     -> 本地经典识别: ClassicFaceRecognizer
     -> 远端识别: HTTP -> server/embedding_service.py
  -> 识别结果标准化
  -> view/main.py 渲染结果与状态
```

### 启动补充链路

```text
应用启动
  -> bootstrap_face_service_tunnel()
  -> MainPage.start_startup_remote_sync()
  -> reconcile_remote_samples()
```

也就是说，SSH 隧道自举和样本同步都被接入了应用生命周期，而不是散落在独立脚本里由界面外部手动驱动。

## 仓库结构

```text
CourseDesign/
├── main.py
├── view/
│   ├── main.py
│   ├── mainDisplay.py
│   └── mainDisplay.ui
├── service/
│   ├── vision.py
│   ├── remote_sync.py
│   ├── ssh_tunnel.py
│   └── recognition_result.py
├── store/
│   └── config.py
├── utils/
│   ├── common.py
│   ├── vision.py
│   ├── mediapipe/
│   └── mindface/
├── server/
│   ├── embedding_service.py
│   ├── backend_state.py
│   ├── classifier_backend.py
│   ├── classifier_assets.py
│   ├── mindspore_backend.py
│   ├── model_assets.py
│   ├── sample_dataset.py
│   └── README.md
├── scripts/
├── tests/
│   ├── service/
│   ├── server/
│   └── scripts/
├── dataset/
├── data/
├── pretrained/
└── temp/
```

## 核心模块职责

### 应用入口与 UI

- `main.py`
  应用装配入口，负责实例化 `ConfigStore`、`VisionService`、`MainPage`，并在退出时清理自动创建的 SSH 隧道。
- `view/main.py`
  主界面控制器。负责界面编排、交互事件绑定、显示帧轮询、日志消费、样本概览刷新，以及启动后的后台远端同步。
- `view/mainDisplay.py` / `view/mainDisplay.ui`
  UI 基础结构，其中持久化布局逻辑主要收敛在 `view/main.py`。

### 本地服务与调度

- `service/vision.py`
  本地运行时核心。内部同时承载：
  - `VisionService`：桌面端服务入口
  - `ReadCameraThread`：摄像头读取与检测分发
  - `ClassicFaceRecognizer`：本地经典检测与识别
  - `RecogThread`：识别线程，统一处理本地/远端识别结果
  - `RegisterThread`：注册线程，处理样本注册与远端提交
  - `MindFaceService`：MindSpore 检测能力接入
- `service/remote_sync.py`
  本地样本目录和远端身份库之间的对账模块，维护同步清单、差异签名、增量重建和删除传播。
- `service/ssh_tunnel.py`
  SSH 隧道管理模块，根据环境变量决定是否自动建立到远端服务的本地端口转发。
- `service/recognition_result.py`
  识别结果归一化的最小适配层，把不同后端返回格式统一成界面可消费的显示名称。

### 共享状态

- `store/config.py`
  维护检测开关、识别开关、检测方法、识别方法等共享配置，并提供线程间使用的 `Queue`。

### 算法与资源

- `utils/vision.py`
  本地图像处理基础能力，包括检测辅助、裁剪和训练数据整理。
- `utils/mediapipe/`
  轻量检测能力封装。
- `utils/mindface/`
  MindSpore 相关检测/识别代码与预训练资源，作为深度路线的算法实现区。
- `pretrained/`
  经典检测器级别的本地资源。
- `dataset/full/`
  桌面端采集得到的本地身份样本主目录，也是远端同步的源数据。
- `temp/`
  本地同步清单等运行期临时产物目录。

### 远端服务

- `server/embedding_service.py`
  服务端主入口，暴露注册、识别、健康检查、身份列表、删除身份和分类器训练相关接口。
- `server/backend_state.py`
  记录服务端当前活跃后端、模型脏状态、训练状态和产物元数据。
- `server/mindspore_backend.py`
  `mindspore_lite` embedding 推理后端，用于远端 embedding 识别。
- `server/classifier_backend.py`
  基于 `mindspore_lite` 的 CNN 分类器推理后端。
- `server/model_assets.py` / `server/classifier_assets.py`
  统一管理服务端模型产物路径。
- `server/sample_dataset.py`
  负责把身份样本整理成训练所需的数据集快照和目录结构。

### 脚本与测试

- `scripts/`
  训练、导出、数据准备和辅助同步脚本所在层，用于支持服务端模型流程和离线数据处理。
- `tests/service/`
  验证桌面端服务的小型工具逻辑。
- `tests/server/`
  覆盖远端 API、后端状态、模型路径和样本集准备逻辑。
- `tests/scripts/`
  覆盖训练前数据准备和模型导出脚本。

## 桌面端与服务端边界

桌面端负责：

- 摄像头接入
- 检测结果叠加显示
- 本地样本采集与删除
- 本地经典模型训练与识别
- 远端识别请求发起
- 启动期同步与运行期状态反馈

远端服务负责：

- 接收裁剪后的人脸图像
- 管理服务端身份样本与 embedding
- 执行 embedding 识别
- 执行 CNN 分类器识别
- 维护训练产物与后端状态

这意味着仓库的核心设计不是“单体脚本堆叠”，而是“桌面调度端 + 可选远端识别服务 + 训练/导出支撑脚本”的分层结构。

## 延伸阅读

- 根目录聚焦整体架构
- `server/README.md` 聚焦远端服务接口与内部职责
