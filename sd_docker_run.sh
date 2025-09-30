#!/bin/bash

WORK_DIR="./deep-e-sd-series/deepE-sd-series"
IMAGE_NAME="flux-app:latest"
CONTAINER_NAME="flux_app_prod"

ORIGINAL_DIR=$(pwd)

if [ ! -d "$WORK_DIR" ]; then
  echo "Error: Directory $WORK_DIR does not exist!"
  exit 1
fi

cd "$WORK_DIR" || exit

if [ "$1" == "down" ]; then
  echo "Stopping and removing container..."
  docker stop "$CONTAINER_NAME" 2>/dev/null
  docker rm "$CONTAINER_NAME" 2>/dev/null
  echo "Container stopped and removed."
  exit 0
fi

echo "Building image..."
docker build -t "$IMAGE_NAME" .

if [ $? -ne 0 ]; then
  echo "Image build failed!"
  exit 1
fi

cd "$ORIGINAL_DIR" || exit

echo "Starting container..."

docker run -d --name "$CONTAINER_NAME" --gpus all -p 5051:5050 \
  -v ./deep-e-sd-series/flux_models:/app/flux_models \
  -v ./imageGeneration:/app/generated_images \
  -v ./deep-e-sd-series/models_config.yaml:/app/models_config.yaml:rw \
  -v ./models:/app/models \
  -e CUDA_VISIBLE_DEVICES=0 -e PYTHONPATH=/app \
  "$IMAGE_NAME"

if [ $? -eq 0 ]; then
  echo "Container started successfully, name: $CONTAINER_NAME"
else
  echo "Container start failed!"
  exit 1
fi