# Flux Docker 部署指南

本项目提供了完整的Docker化解决方案，用于部署Flux图像生成应用。

## 🚀 快速开始

### 前置要求

1. **Docker & Docker Compose**
   ```bash
   # 安装Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # 安装Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **NVIDIA Docker支持**
   ```bash
   # 安装NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

### 一键部署

```bash
# 运行构建和部署脚本
./build_and_run.sh
```

### 手动部署

1. **构建镜像**
   ```bash
   docker build -t flux-app:latest .
   ```

2. **启动服务**
   ```bash
   docker-compose up -d
   ```

3. **查看状态**
   ```bash
   docker-compose ps
   docker-compose logs -f flux-app
   ```

## 📁 目录结构

```
flux_docker/
├── Dockerfile              # Docker镜像构建文件
├── docker-compose.yml      # Docker Compose配置
├── build_and_run.sh        # 一键部署脚本
├── requirements.txt        # Python依赖
├── dynamic_flux_app.py     # 主应用程序
├── model_manager.py        # 模型管理器
├── flux_models/            # 模型文件目录（挂载到宿主机）
└── generated_images/       # 生成图片目录（挂载到宿主机）
```

## 🔧 配置说明

### 挂载点

- **模型文件夹**: `./flux_models` → `/app/flux_models`
- **生成图片**: `./generated_images` → `/app/generated_images`

### 端口映射

- **应用端口**: `5050:5050`
- **访问地址**: http://localhost:5050

### 环境变量

- `CUDA_VISIBLE_DEVICES=0`: 指定GPU设备
- `PYTHONPATH=/app`: Python路径配置

## 🛠️ 常用命令

### 服务管理

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看日志
docker-compose logs -f flux-app

# 查看服务状态
docker-compose ps
```

### 容器管理

```bash
# 进入容器
docker-compose exec flux-app bash

# 查看容器资源使用
docker stats flux-docker-app

# 重新构建镜像
docker-compose build --no-cache
```

## 🔍 API端点

- **健康检查**: `GET /health`
- **模型列表**: `GET /models`
- **文本生图**: `POST /text2image`
- **图片生图**: `POST /image2image`
- **任务状态**: `GET /task/<task_id>`

## 📊 监控和调试

### 健康检查

```bash
curl http://localhost:5050/health
```

### 查看GPU使用情况

```bash
# 在容器内执行
docker-compose exec flux-app nvidia-smi
```

### 查看应用日志

```bash
# 实时日志
docker-compose logs -f flux-app

# 最近100行日志
docker-compose logs --tail=100 flux-app
```

## 🚨 故障排除

### 常见问题

1. **GPU不可用**
   - 检查NVIDIA驱动安装
   - 确认nvidia-docker2已安装
   - 验证GPU在容器中可见：`docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

2. **内存不足**
   - 调整Docker内存限制
   - 检查模型大小和GPU内存

3. **端口冲突**
   - 修改docker-compose.yml中的端口映射
   - 检查端口是否被占用：`netstat -tlnp | grep 5050`

4. **模型加载失败**
   - 检查flux_models目录中的模型文件
   - 确认模型路径配置正确

### 日志分析

```bash
# 查看错误日志
docker-compose logs flux-app | grep ERROR

# 查看警告日志
docker-compose logs flux-app | grep WARNING
```

## 🔄 更新和维护

### 更新应用

```bash
# 停止服务
docker-compose down

# 重新构建镜像
docker build -t flux-app:latest .

# 启动服务
docker-compose up -d
```

### 清理资源

```bash
# 清理未使用的镜像
docker image prune -f

# 清理未使用的容器
docker container prune -f

# 清理未使用的卷
docker volume prune -f
```

## 📝 注意事项

1. **模型文件**: 确保flux_models目录包含所需的模型文件
2. **GPU内存**: 根据模型大小调整GPU内存分配
3. **存储空间**: 生成的图片会占用存储空间，定期清理
4. **网络访问**: 首次运行可能需要下载依赖，确保网络连接正常
5. **权限问题**: 确保挂载目录有正确的读写权限

## 🆘 获取帮助

如果遇到问题，请检查：

1. Docker和NVIDIA Docker是否正确安装
2. GPU驱动是否兼容
3. 模型文件是否完整
4. 网络连接是否正常
5. 系统资源是否充足

---

**祝您使用愉快！** 🎉