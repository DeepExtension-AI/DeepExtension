#!/bin/bash

# Run preparation script
SYSTEM=$(uname -s)
source prod.env
if [ "$SYSTEM" = "Darwin" ]; then
  ReplaceCommand="sed -i ''"
  MacTrainingHost="host.docker.internal"
elif [ "$SYSTEM" = "Linux" ]; then
  ReplaceCommand='sed -i'
  MacTrainingHost=${SCP_GO_AI_TRAINING_HOST}
elif [[ "$SYSTEM" =~ ^(MINGW|MSYS|CYGWIN) ]]; then
  ReplaceCommand='sed -i'
  MacTrainingHost=${SCP_GO_AI_TRAINING_HOST}
fi

# Configuration file paths
CONF_FILE="custom.conf"
ENV_FILE=".env"
> "$ENV_FILE"
## Copy prod.env content to .env (including image.env content)
cat prod.env > "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Extract configuration values
db_host=$(grep -E '^DB_HOST=' "$CONF_FILE" | cut -d'=' -f2-)
db_port=$(grep -E '^DB_PORT=' "$CONF_FILE" | cut -d'=' -f2-)
db_user=$(grep -E '^DB_USER=' "$CONF_FILE" | cut -d'=' -f2-)
db_pass=$(grep -E '^DB_PASS=' "$CONF_FILE" | cut -d'=' -f2-)
db_name=$(grep -E '^DB_NAME=' "$CONF_FILE" | cut -d'=' -f2-)
with_ai_image=$(grep -E '^WITH_AI_IMAGE=' "$CONF_FILE" | cut -d'=' -f2-)
ui_port=$(grep -E '^UI_AI_EXPOSED_PORT=' "$CONF_FILE" | cut -d'=' -f2-)
redis_used_by_py=$(grep -E '^AI_PY_REDIS_EXPOSED_PORT=' "$CONF_FILE" | cut -d'=' -f2-)
# Set default values if empty
if [ -z "$with_ai_image" ]; then
    echo "WITH_AI_IMAGE Using default value TRUE for Linux environment"
    with_ai_image="true"
fi
# Check the default port
is_port_free() {
    local port=$1
    if (lsof -i :$port >/dev/null 2>&1 || nc -z 127.0.0.1 $port >/dev/null 2>&1); then
        return 1  # in use
    else
        return 0
    fi
}
source prod.env
if [ -z "$ui_port" ]; then
    ui_port=$UI_AI_EXPOSED_PORT
fi
while ! is_port_free $ui_port; do
    echo "Port $ui_port is in use, trying next port..."
    ((ui_port++))
done
echo "UI_AI_EXPOSED_PORT Using free port: $ui_port"
if [ -z "$redis_used_by_py" ]; then
    redis_used_by_py=$AI_PY_REDIS_EXPOSED_PORT
fi
while ! is_port_free $redis_used_by_py; do
    echo "Port $redis_used_by_py is in use, trying next port..."
    ((redis_used_by_py++))
done
echo "AI_PY_REDIS_EXPOSED_PORT Using free port: $redis_used_by_py"


# Validate required database parameters
use_db="false"

# 检查是否需要使用compose创建的数据库
if [ -z "$db_host" ]; then
    ## 使用compose创建的数据库
    echo "We'll use database created by compose"
    use_db="true"
    db_host="postgres-deepe-prod"
    db_port=5432
    db_user=postgres
    db_pass=postgres
    db_name=postgres
else
    ## 检查外部数据库配置是否完整
    missing_fields=()
    [ -z "$db_host" ] && missing_fields+=("DB_HOST")
    [ -z "$db_port" ] && missing_fields+=("DB_PORT")
    [ -z "$db_user" ] && missing_fields+=("DB_USER")
    [ -z "$db_pass" ] && missing_fields+=("DB_PASS")
    [ -z "$db_name" ] && missing_fields+=("DB_NAME")

    if [ ${#missing_fields[@]} -gt 0 ]; then
        echo "错误：以下数据库配置信息不完整，请填写完整信息："
        printf '%s\n' "${missing_fields[@]}"
        exit 1
    fi

    # 如果使用外部数据库，确保use_db=false
    use_db="false"
    echo "将使用外部数据库配置："
    echo "DB_HOST=$db_host"
    echo "DB_PORT=$db_port"
    echo "DB_USER=$db_user"
    echo "DB_NAME=$db_name"
    # 注意：出于安全考虑，不打印密码
fi

check_migrations_applied() {
    echo "检查当前数据库迁移版本..."
    local version=$(docker run --rm -v $(pwd)/migrations:/migrations migrate/migrate \
        -path=/migrations/ \
        -database "postgres://$db_user:$db_pass@$db_host:$db_port/$db_name?sslmode=disable" version 2>/dev/null)

    if [ -n "$version" ] && [ "$version" != "nil" ]; then
        echo "数据库已执行迁移（当前版本: $version），跳过迁移"
        return 0
    else
        echo "数据库未执行过迁移或版本为 nil，需要执行迁移"
        return 1
    fi
}
#  Assign DB_* values to other variables
{
  echo "#db_data"
  echo "DB_HOST=$db_host"
  echo "DB_PORT=$db_port"
  echo "DB_USER=$db_user"
  echo "DB_PASS=$db_pass"
  echo "DB_NAME=$db_name"
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
  echo "SCP_GO_AI_TRAINING_PORT=${TRAINING_START_PORT}"
} >> "$ENV_FILE"

# Handle image version overrides
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

# Final environment variable replacements
$ReplaceCommand "s,^SCP_GO_AI_TRAINING_PORT=.*,SCP_GO_AI_TRAINING_PORT=${TRAINING_START_PORT}," ".env"
$ReplaceCommand "s,^SCP_GO_AI_TRAINING_HOST=.*,SCP_GO_AI_TRAINING_HOST=${MacTrainingHost}," ".env"
$ReplaceCommand "s,^AI_PY_REDIS_EXPOSED_PORT=.*,AI_PY_REDIS_EXPOSED_PORT=$redis_used_by_py," ".env"
$ReplaceCommand "s,^UI_AI_EXPOSED_PORT=.*,UI_AI_EXPOSED_PORT=$ui_port," ".env"
echo $db_host
$ReplaceCommand "s,^DB_HOST=.*,DB_HOST=$db_host," ".env"
$ReplaceCommand "s,^DB_PORT=.*,DB_PORT=$db_port," ".env"
$ReplaceCommand "s,^DB_USER=.*,DB_USER=$db_user," ".env"
$ReplaceCommand "s,^DB_PASS=.*,DB_PASS=$db_pass," ".env"
$ReplaceCommand "s,^DB_NAME=.*,DB_NAME=$db_name," ".env"
source "$ENV_FILE"

# Docker operations
FileLocation="./"
PROJECT_NAME=deepe-prod
# 统一处理函数
handle_compose() {
    local profile=""
    [ "$with_ai_image" = "true" ] && profile="--profile gpu"
    [ "$use_db" = "true" ] && profile="$profile --profile use-db"

    case "$1" in
        "stop")
            docker compose $profile -p "${PROJECT_NAME}" -f ./docker-compose.yml stop || exit 1
            ;;
        "down")
            docker compose $profile -p "${PROJECT_NAME}" -f ./docker-compose.yml down || exit 1
            ;;
        *)
            docker compose $profile -p "${PROJECT_NAME}" -f ./docker-compose.yml up -d --remove-orphans || exit 1
            check_non_running_containers
            [ "$SYSTEM" = "Darwin" ] && start_pm2_app
            ;;
    esac
}

# 检查非运行容器
check_non_running_containers() {
    non_running=$(docker ps --filter "label=com.docker.compose.project=$PROJECT_NAME" \
                 --format "{{.ID}} {{.Names}} {{.Status}}" | grep -v "Up ")

    if [ -n "$non_running" ]; then
        echo "Found non-running containers:"
        echo "$non_running"
        exit 1
    else
        echo "All containers are running normally"
    fi
}

# macOS专用PM2启动
start_pm2_app() {
    APP_NAME="training-py"
    APP_SCRIPT="./app.py"

    cd deep-e-python || exit 1
    if pm2 list | grep -q "$APP_NAME"; then
        echo "🔄 Restarting $APP_NAME..."
        pm2 delete "$APP_NAME"
    else
        echo "🚀 Starting $APP_NAME..."
    fi

    pm2 start "$APP_SCRIPT" --name "$APP_NAME"
    pm2 save
}

# 主逻辑
if [ "$SYSTEM" = "Darwin" ]; then
    with_ai_image="false"
    handle_compose "$1"
elif [[ "$SYSTEM" =~ ^(Linux|MINGW|MSYS|CYGWIN)$ ]]; then
    handle_compose "$1"
else
    echo "Unsupported system: $SYSTEM"
    exit 1
fi