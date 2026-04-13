# 远端识别服务模块

`server/` 目录用于承载可独立部署的远端识别服务。它的职责不是替代桌面端，而是把推理与身份库维护能力从 GUI 程序中拆出来，便于后续单独部署、调试和扩展。

## 模块定位

桌面端主要负责：

- 摄像头采集
- 本地样本管理
- 界面展示
- 本地训练与识别流程调度

`server/` 主要负责：

- 接收上传的人脸图片
- 执行远端识别推理
- 维护远端身份样本与统计信息
- 返回识别结果给桌面端

## 接口结构

当前服务提供以下接口：

- `POST /register`
  用于注册单张人脸样本。
- `POST /recognize`
  用于提交单张人脸图并返回识别结果。
- `GET /health`
  用于查看服务存活状态和运行信息。
- `GET /identities`
  用于查询当前身份库概况。
- `POST /delete_identity`
  用于删除指定身份。

## 目录职责

```text
server/
├── embedding_service.py
├── README.md
└── runtime/
    └── .gitkeep
```

运行过程中，服务会把运行时数据写入 `server/runtime/`，例如：

- 样本图片
- 识别用数据文件
- 统计信息

这些运行期数据不应提交到仓库。

## 与桌面端关系

桌面端通过 `service/vision.py` 中的识别与注册线程调用远端服务接口。也就是说：

1. 桌面端完成人脸裁剪。
2. 裁剪结果通过 HTTP 提交给远端服务。
3. 远端服务完成识别或注册处理。
4. 结果返回给桌面端界面。

这种设计可以把 GUI、摄像头处理和远端推理解耦，便于后续替换服务实现。

## 运行依赖

当前服务侧实现基于：

- `FastAPI`
- `uvicorn`
- `mindspore_lite`
- `PyTorch`
- `facenet-pytorch`

## 模型与后端

服务以 `InceptionResnetV1(vggface2)` 作为主要的模型转换目标：

- 优先使用 `mindspore_lite` 转换后的推理模型进行识别
- 若转换模型不可用，则回退到现有的 PyTorch embedding 服务实现

模型转换使用随仓库提供的 `converter_lite`，转换完成后建议用 `benchmark` 进行一致性验证。

这里保留的是服务模块本身，不绑定特定机器、特定端口或特定部署环境。实际部署时，请根据自己的服务器环境单独配置。

## 建议阅读顺序

如果你要继续修改远端服务，建议按下面顺序阅读：

1. `server/embedding_service.py`
2. `service/vision.py`
3. `service/remote_sync.py`

这样可以先看远端服务本身，再看桌面端如何发请求和做本地远端同步。
