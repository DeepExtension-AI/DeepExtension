```bash
#!/bin/bash
set -e

# 镜像构建目录（替换成你的Dockerfile所在目录）
BUILD_DIR="./deep-e-sd-series/deepE-sd-series"

# 镜像名称
IMAGE_NAME="flux-app:latest"

# 构建镜像
echo "Enter the build directory: $BUILD_DIR"
cd "$BUILD_DIR"
echo "Start building the mirror image: $IMAGE_NAME"
DOCKER_BUILDKIT=0 docker build -t "$IMAGE_NAME" .

# 回到执行 run 的目录
cd -

# 运行容器（你可以按需替换 run 命令参数）
echo "Start the container..."

docker run -d --name flux_app_prod --gpus all -p 5051:5050 \
  -v ./deep-e-sd-series/flux_models:/app/flux_models \
  -v ./imageGeneration:/app/generated_images \
  -v ./deep-e-sd-series/models_config.yaml:/app/models_config.yaml:rw \
  -v ./models:/app/models \
  -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/app \
  "$IMAGE_NAME"
```
