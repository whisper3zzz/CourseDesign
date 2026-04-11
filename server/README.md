# Embedding Service

这个目录提供一个可独立部署的人脸 embedding 后端，直接兼容当前桌面程序已有的接口形态：

- `POST /register`
- `POST /recognize`
- `GET /health`

## 设计目标

当前桌面程序已经会在本地完成人脸裁剪，因此这个后端默认假设上传过来的图片已经是单张人脸图。

服务端职责只保留三件事：

- 提取 embedding
- 维护本地 embedding 库
- 完成人脸比对并返回姓名或“未知”

## 运行环境

推荐部署到 `Linux x86_64`。

默认实现使用：

- `FastAPI`
- `PyTorch`
- `facenet-pytorch`

之所以先不用 `MindSpore` 作为在线服务底座，是因为在线部署优先级是“先稳定跑通”，而不是把服务端环境锁死在更挑系统版本的依赖上。

## 目录结构

运行后会在 `server/runtime/` 下生成数据：

```text
server/runtime/
├── faces/
│   └── 张三/
├── embeddings/
│   └── 张三.npy
└── metadata/
    └── stats.json
```

## 接口说明

### `GET /health`

返回服务状态、设备信息和当前 embedding 库规模。

### `POST /register`

表单参数：

- `name`
- `photo`

行为：

- 保存上传图片
- 提取 embedding
- 追加到对应身份的人脸库

### `POST /recognize`

表单参数：

- `photo`

返回示例：

```json
{
  "name": "张三",
  "matched": true,
  "score": 0.8421,
  "threshold": 0.72,
  "identity_count": 5,
  "embedding_count": 18
}
```

未命中时：

```json
{
  "name": "未知",
  "matched": false,
  "score": 0.5512,
  "threshold": 0.72,
  "identity_count": 5,
  "embedding_count": 18
}
```

## 启动方式

### 1. 用 `uv` 创建环境

```bash
uv venv .service-venv
```

### 2. 安装依赖

如果直接用 `server/requirements.txt` 能成功解析，最简单：

```bash
uv pip install --python .service-venv/bin/python -r server/requirements.txt
```

如果目标机需要先固定 CPU 版 `PyTorch`，可以用下面这组已经验证过的命令：

```bash
uv pip install --python .service-venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.2.2 torchvision==0.17.2

uv pip install --python .service-venv/bin/python \
  fastapi "uvicorn[standard]" python-multipart facenet-pytorch==2.6.0
```

### 3. 启动服务

前台启动：

```bash
./.service-venv/bin/python -m uvicorn server.embedding_service:app --host 0.0.0.0 --port 8000
```

后台启动：

```bash
mkdir -p logs
nohup ./.service-venv/bin/python -m uvicorn server.embedding_service:app \
  --host 0.0.0.0 \
  --port 8000 > logs/embedding_service.log 2>&1 &
```

## 可选环境变量

- `FACE_SERVICE_DATA_DIR`
  默认 `server/runtime`
- `FACE_SERVICE_MODEL`
  默认 `vggface2`
- `FACE_SERVICE_THRESHOLD`
  默认 `0.72`
- `FACE_SERVICE_DEVICE`
  默认自动选择 `cuda` 或 `cpu`

## 与当前桌面程序对接

桌面端现在支持通过环境变量指定服务地址：

```bash
FACE_SERVICE_BASE_URL=http://<server-host>:8000 uv run python main.py
```

如果服务端 `8000` 没有直接对外暴露，推荐在桌面端机器上先建立 SSH 隧道：

```bash
ssh -o StrictHostKeyChecking=no -L 18000:127.0.0.1:8000 <user>@<server-host> -p <ssh-port> -N
```

然后直接启动桌面程序即可。当前桌面端默认会优先请求：

```bash
http://127.0.0.1:18000
```

## 健康检查

```bash
curl http://127.0.0.1:8000/health
```

如果返回类似下面的数据，说明服务已经跑起来：

```json
{
  "status": "ok",
  "device": "cpu",
  "model": "vggface2",
  "threshold": 0.72,
  "identity_count": 0,
  "embedding_count": 0
}
```
