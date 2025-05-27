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
fi

# Configuration file paths
CONF_FILE="custom.conf"
ENV_FILE=".env"
> "$ENV_FILE"

## Copy prod.env content to .env (including image.env content)
cat prod.env > "$ENV_FILE"
echo "" >> "$ENV_FILE"

# 1. Check if custom.conf exists
if [ ! -f "$CONF_FILE" ]; then
    echo "Error: Configuration file $CONF_FILE not found"
    exit 1
fi

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
if [ -z "$db_host" ]; then
    db_host="host.docker.internal"
fi
if [ -z "$with_ai_image" ]; then
    echo "WITH_AI_IMAGE Using default value TRUE for Linux environment"
    with_ai_image="true"
fi
if [ -z "$ui_port" ]; then
    echo "UI_AI_EXPOSED_PORT Using default value 88"
    ui_port=88
fi
if [ -z "$redis_used_by_py" ]; then
    echo "AI_PY_REDIS_EXPOSED_PORT Using default value 6490"
    redis_used_by_py=6490
fi

# Validate required database parameters
if [ -z "$db_port" ] || [ -z "$db_user" ] || [ -z "$db_pass" ] || [ -z "$db_name" ]; then
    echo "Error: DB_PORT or DB_USER or DB_PASS or DB_NAME is [not set] or [empty]"
    exit 1
fi

# Append custom.conf to .env
cat "$CONF_FILE" >> "$ENV_FILE"

# 4. Assign DB_* values to other variables
{
  echo "#db_data"
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
  echo "UI_AI_EXPOSED_PORT=$ui_port"
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
source "$ENV_FILE"

# Docker operations
FileLocation="./"
PROJECT_NAME=deepe-prod

if [ "$SYSTEM" = "Darwin" ]; then
  # MacOS specific operations
  docker compose -p "${PROJECT_NAME}" -f ./docker-compose.yml --env-file ./.env up -d --remove-orphans || exit 1

  # Check for non-running containers
  non_running=$(docker ps --filter "label=com.docker.compose.project=$PROJECT_NAME" --format "{{.ID}} {{.Names}} {{.Status}}" | grep -v "Up ")

  if [ -n "$non_running" ]; then
    echo "Found non-running containers:"
    echo "$non_running"
    exit 1  # Exit with code 1 if non-running containers exist
  else
    echo "$non_running"
    echo "All containers are running normally"
  fi

  # PM2 process management for Python app
  APP_NAME="training-py"
  APP_SCRIPT="./app.py"

  ## Requires pm2 installed: npm install pm2 -g
  ## To view Python container logs: pm2 logs training-py

  # Check if service exists
  export TRAINING_START_PORT=$TRAINING_START_PORT
  export AI_PY_REDIS_EXPOSED_PORT=$AI_PY_REDIS_EXPOSED_PORT
  cd deep-e-python || exit 1
  if pm2 list | grep -q "$APP_NAME"; then
      echo "🔄 Restarting $APP_NAME..."
      pm2 delete "$APP_NAME"
      pm2 start "$APP_SCRIPT" --name "$APP_NAME"
  else
      echo "🚀 Starting $APP_NAME..."
      pm2 start "$APP_SCRIPT" --name "$APP_NAME"
  fi
  pm2 save
elif [ "$SYSTEM" = "Linux" ]; then
  # Linux specific operations
  if [ "$with_ai_image" = "true" ]; then
    docker compose --profile gpu -p "${PROJECT_NAME}" -f ./docker-compose.yml up -d --remove-orphans || exit 1
  else
    docker compose -p "${PROJECT_NAME}" -f ./docker-compose.yml up -d --remove-orphans || exit 1
  fi

  # Check for non-running containers
  non_running=$(docker ps --filter "label=com.docker.compose.project=$PROJECT_NAME" --format "{{.ID}} {{.Names}} {{.Status}}" | grep -v "Up ")

  if [ -n "$non_running" ]; then
    echo "Found non-running containers:"
    echo "$non_running"
    exit 1  # Exit with code 1 if non-running containers exist
  else
    echo "$non_running"
    echo "All containers are running normally"
  fi
fi