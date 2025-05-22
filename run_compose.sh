#!/bin/bash

# 运行准备脚本
SYSTEM=$(uname -s)
if [ "$SYSTEM" = "Darwin" ]; then
  ReplaceCommand="sed -i ''"
elif [ "$SYSTEM" = "Linux" ]; then
  ReplaceCommand='sed -i'
fi

# 配置文件路径
CONF_FILE="custom.conf"
ENV_FILE=".env"
> "$ENV_FILE"
source prod.env
$ReplaceCommand  "s,^SCP_GO_AI_TRAINING_PORT=.*,SCP_GO_AI_TRAINING_PORT=${TRAINING_START_PORT}," "prod.env"
## 复制prod文件给env (包括image.env的内容)
cat prod.env > "$ENV_FILE"
echo "" >> "$ENV_FILE"

# 1. 检查 custom.conf 是否存在
if [ ! -f "$CONF_FILE" ]; then
    echo "错误：配置文件 $CONF_FILE 不存在"
    exit 1
fi

db_host=$(grep -E '^DB_HOST=' "$CONF_FILE" | cut -d'=' -f2-)
db_port=$(grep -E '^DB_PORT=' "$CONF_FILE" | cut -d'=' -f2-)
db_user=$(grep -E '^DB_USER=' "$CONF_FILE" | cut -d'=' -f2-)
db_pass=$(grep -E '^DB_PASS=' "$CONF_FILE" | cut -d'=' -f2-)
db_name=$(grep -E '^DB_NAME=' "$CONF_FILE" | cut -d'=' -f2-)

if [ -z "$db_host" ] || [ -z "$db_port" ]; then
    echo "错误：DB_HOST 或 DB_PORT 未设置或值为空"
    exit 1
fi

cat "$CONF_FILE" >> "$ENV_FILE"

# 4. 将 DB_* 赋值给其他变量
{
  echo "SCP_GO_AI_DB_HOST=$db_host"
  echo "CRON_AI_DB_HOST=$db_host"
  echo "DEEPE_RAG_DB_HOST=$db_host"
  echo "SCP_GO_AI_DB_PORT=$db_port"
  echo "CRON_AI_DB_PORT=$db_port"
  echo "DEEPE_RAG_DB_PORT=$db_port"
  echo "SCP_GO_AI_DB_USERNAME=$db_user"
  echo "CRON_AI_DB_USERNAME=$db_user"
  echo "DEEPE_RAG_DB_USERNAME=$db_user"
  echo "SCP_GO_AI_DB_PASSWORD=$db_pass"
  echo "CRON_AI_DB_PASSWORD=$db_pass"
  echo "DEEPE_RAG_DB_PASSWORD=$db_pass"
  echo "SCP_GO_AI_DB_NAME=$db_name"
  echo "CRON_AI_DB_NAME=$db_name"
  echo "DEEPE_RAG_DB_NAME=$db_name"
} >> "$ENV_FILE"

if [ -f "image.env" ]; then
  source image.env

  # Helper function to handle replacement
  replace_var() {
    local var_name=$1
    local default_val=$2
    local custom_val=$3
    local value=${custom_val:-$default_val}
    $ReplaceCommand "s,^${var_name}=.*,${var_name}=${value}," "$ENV_FILE"
  }

  replace_var "SCP_AI_IMAGE_VERSION" "$SCP_AI_IMAGE_VERSION" "$CUSTOM_DEEP_E_BACKEND_IMAGE_VERSION"
  replace_var "SCP_AI_IMAGE_NAME" "$SCP_AI_IMAGE_NAME" "$CUSTOM_DEEP_E_BACKEND_IMAGE_NAME"
  replace_var "UI_AI_IMAGE_VERSION" "$UI_AI_IMAGE_VERSION" "$CUSTOM_DEEP_E_UI_IMAGE_VERSION"
  replace_var "UI_AI_IMAGE_NAME" "$UI_AI_IMAGE_NAME" "$CUSTOM_DEEP_E_UI_IMAGE_NAME"
  replace_var "CRON_AI_IMAGE_VERSION" "$CRON_AI_IMAGE_VERSION" "$CUSTOM_DEEP_E_CRON_IMAGE_VERSION"
  replace_var "CRON_AI_IMAGE_NAME" "$CRON_AI_IMAGE_NAME" "$CUSTOM_DEEP_E_CRON_IMAGE_NAME"
  replace_var "TRAINING_AI_IMAGE_NAME" "$TRAINING_AI_IMAGE_NAME" "$CUSTOM_DEEP_E_PYTHON_IMAGE_NAME"
  replace_var "TRAINING_AI_IMAGE_VERSION" "$TRAINING_AI_IMAGE_VERSION" "$CUSTOM_DEEP_E_PYTHON_IMAGE_VERSION"
  replace_var "RAG_IMAGE_VERSION" "$RAG_IMAGE_VERSION" "$CUSTOM_DEEP_E_RAG_IMAGE_VERSION"
  replace_var "RAG_IMAGE_NAME" "$RAG_IMAGE_NAME" "$CUSTOM_DEEP_E_RAG_IMAGE_NAME"
fi


echo "" >> "$ENV_FILE"
source "$ENV_FILE"

FileLocation="./"
PROJECT_NAME=scp-ai-prod

if [ "$SYSTEM" = "Darwin" ]; then
  docker-compose -p "${PROJECT_NAME}" -f ./docker-compose.yml --env-file ./.env up -d --remove-orphans
  APP_NAME="training-py"
  APP_SCRIPT="./app.py"

  ## 需要提前安装pm2: npm install pm2 -g
  ## 查看Python容器日志 pm2 logs training-py
  # 检查服务是否存在
  cd deep-e-python || exit 1
  if pm2 list | grep -q "$APP_NAME"; then
      echo "🔄 Restarting $APP_NAME..."
      pm2 restart "$APP_NAME"
  else
      echo "🚀 Starting $APP_NAME..."
      pm2 start "$APP_SCRIPT" --name "$APP_NAME"
  fi
  pm2 save
elif [ "$SYSTEM" = "Linux" ]; then
  if [ "$WITH_AI_IMAGE" = "true" ]; then
    docker-compose -p "${PROJECT_NAME}" -f ./docker-compose.yml up -d --remove-orphans --profile gpu
  else
    docker-compose -p "${PROJECT_NAME}" -f ./docker-compose.yml up -d --remove-orphans
  fi
fi