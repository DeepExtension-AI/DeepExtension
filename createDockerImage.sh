#!/bin/bash

# 定义Dockerfile文件的路径
dockerfile_path1="./scp-go/Dockerfile"
dockerfile_path2="./scp-ui/Dockerfile"

# 执行docker build命令
docker build -f $dockerfile_path1 -t scp-go .
docker build -f $dockerfile_path2 -t scp-ui .