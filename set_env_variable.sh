#!/bin/bash
ReplaceCommand=$(echo $1 | sed 's/_/ /g')
echo "CurrentCommand:$ReplaceCommand"
echo "CurrentFileLocation:$2"
#处理镜像
if [ -f "$2image.env" ]; then
  source $2image.env
else
  echo "无法更新镜像版本！"
fi

# 默认环境变量值
if [ -f "$2.env" ]; then
      echo "正在加载文件:$2.env:"$2.env
      source $2.env
      PracticalEnv=$2.env
      # 验证环境变量是否已加载
      echo "当前环境: $GLOBAL_ENV"
      DEFAULT_ENV=$GLOBAL_ENV
else
  echo ".env file not found!"
  exit 1
fi
EE=$(echo $DEFAULT_ENV | cut -d '.' -f 1)
FileLocation="$2environment"
ENV="$FileLocation/$DEFAULT_ENV"   ## ./environment/local.env
case $ENV in
  $FileLocation/local.env)
    if [ -f "$FileLocation/local.env" ]; then
      source $FileLocation/local.env
    else
      echo "local.env file not found!"
      exit 1
    fi
  ;;
  $FileLocation/dev.env)
    if [ -f "$FileLocation/dev.env" ]; then
      source $FileLocation/dev.env
    else
      echo "dev.env file not found!"
      exit 1
    fi
  ;;
  $FileLocation/auto_test.env)
    if [ -f "$FileLocation/auto_test.env" ]; then
      source $FileLocation/auto_test.env
    else
      echo "auto_test.env file not found!"
      exit 1
    fi
  ;;
  $FileLocation/devpre.env)
    if [ -f "$FileLocation/devpre.env" ]; then
      source $FileLocation/devpre.env
    else
      echo "devpre.env file not found!"
      exit 1
    fi
  ;;
  $FileLocation/prod.env)
    if [ -f "$FileLocation/prod.env" ]; then
      source $FileLocation/prod.env
    else
      echo "prod.env file not found!"
      exit 1
    fi
  ;;
  *)
    echo "Unknown environment: $ENV"
    exit 1
    ;;
esac
PracticalEnv=.env
# 处理自定义镜像名称和版本：非空则将image.env文件变量替换给.env文件指定变量
if [ -z "$CUSTOM_SCP_AI_IMAGE_VERSION" ]; then
	$ReplaceCommand  "s,^SCP_AI_IMAGE_VERSION=.*,SCP_AI_IMAGE_VERSION=${SCP_AI_IMAGE_VERSION}," "$PracticalEnv"
else
	$ReplaceCommand  "s,^SCP_AI_IMAGE_VERSION=.*,SCP_AI_IMAGE_VERSION=${CUSTOM_SCP_AI_IMAGE_VERSION}," "$PracticalEnv"
fi

if [ -z $CUSTOM_SCP_AI_IMAGE_NAME ]; then
	$ReplaceCommand  "s,^SCP_AI_IMAGE_NAME=.*,SCP_AI_IMAGE_NAME=${SCP_AI_IMAGE_NAME}," "$PracticalEnv"
else
	$ReplaceCommand  "s,^SCP_AI_IMAGE_NAME=.*,SCP_AI_IMAGE_NAME=${CUSTOM_SCP_AI_IMAGE_NAME}," "$PracticalEnv"
fi

if [ -z $CUSTOM_UI_AI_IMAGE_VERSION ]; then
	$ReplaceCommand  "s,^UI_AI_IMAGE_VERSION=.*,UI_AI_IMAGE_VERSION=${UI_AI_IMAGE_VERSION}," "$PracticalEnv"
else
	$ReplaceCommand  "s,^UI_AI_IMAGE_VERSION=.*,UI_AI_IMAGE_VERSION=${CUSTOM_UI_AI_IMAGE_VERSION}," "$PracticalEnv"
fi

if [ -z $CUSTOM_UI_AI_IMAGE_NAME ]; then
	$ReplaceCommand  "s,^UI_AI_IMAGE_NAME=.*,UI_AI_IMAGE_NAME=${UI_AI_IMAGE_NAME}," "$PracticalEnv"
else
	$ReplaceCommand  "s,^UI_AI_IMAGE_NAME=.*,UI_AI_IMAGE_NAME=${CUSTOM_UI_AI_IMAGE_NAME}," "$PracticalEnv"
fi

if [ -z $CUSTOM_CRON_AI_IMAGE_VERSION ]; then
	$ReplaceCommand  "s,^CRON_AI_IMAGE_VERSION=.*,CRON_AI_IMAGE_VERSION=${CRON_AI_IMAGE_VERSION}," "$PracticalEnv"
else
	$ReplaceCommand  "s,^CRON_AI_IMAGE_VERSION=.*,CRON_AI_IMAGE_VERSION=${CUSTOM_CRON_AI_IMAGE_VERSION}," "$PracticalEnv"
fi
if [ -z $CUSTOM_CRON_AI_IMAGE_NAME ]; then
	$ReplaceCommand  "s,^CRON_AI_IMAGE_NAME=.*,CRON_AI_IMAGE_NAME=${CRON_AI_IMAGE_NAME}," "$PracticalEnv"
else
	$ReplaceCommand  "s,^CRON_AI_IMAGE_NAME=.*,CRON_AI_IMAGE_NAME=${CUSTOM_CRON_AI_IMAGE_NAME}," "$PracticalEnv"
fi
if [ -z $CUSTOM_TRAINING_AI_IMAGE_NAME ]; then
	$ReplaceCommand  "s,^TRAINING_AI_IMAGE_NAME=.*,TRAINING_AI_IMAGE_NAME=${TRAINING_AI_IMAGE_NAME}," "$PracticalEnv"
else
	$ReplaceCommand  "s,^TRAINING_AI_IMAGE_NAME=.*,TRAINING_AI_IMAGE_NAME=${CUSTOM_TRAINING_AI_IMAGE_NAME}," "$PracticalEnv"
fi
if [ -z $CUSTOM_TRAINING_AI_IMAGE_VERSION ]; then
	$ReplaceCommand  "s,^TRAINING_AI_IMAGE_VERSION=.*,TRAINING_AI_IMAGE_VERSION=${TRAINING_AI_IMAGE_VERSION}," "$PracticalEnv"
else
	$ReplaceCommand  "s,^TRAINING_AI_IMAGE_VERSION=.*,TRAINING_AI_IMAGE_VERSION=${CUSTOM_TRAINING_AI_IMAGE_VERSION}," "$PracticalEnv"
fi
# 处理第三方镜像名称和版本：非空则将environment.env文件变量替换给.env文件指定变量
$ReplaceCommand  "s,^AI_CONSUL_IMAGE_VERSION=.*,AI_CONSUL_IMAGE_VERSION=${AI_CONSUL_IMAGE_VERSION}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_REDIS_IMAGE_VERSION=.*,AI_REDIS_IMAGE_VERSION=${AI_REDIS_IMAGE_VERSION}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_ES_IMAGE_VERSION=.*,AI_ES_IMAGE_VERSION=${AI_ES_IMAGE_VERSION}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_FB_IMAGE_VERSION=.*,AI_FB_IMAGE_VERSION=${AI_FB_IMAGE_VERSION}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_KIBANA_IMAGE_VERSION=.*,AI_KIBANA_IMAGE_VERSION=${AI_KIBANA_IMAGE_VERSION}," "$PracticalEnv"
# 处理对外暴露端口：非空则将environment.env文件变量替换给.env文件指定变量
$ReplaceCommand  "s,^AI_CONSUL_HTTP_EXPOSED_PORT=.*,AI_CONSUL_HTTP_EXPOSED_PORT=${AI_CONSUL_HTTP_EXPOSED_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_CONSUL_HTTP2_EXPOSED_PORT=.*,AI_CONSUL_HTTP2_EXPOSED_PORT=${AI_CONSUL_HTTP2_EXPOSED_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_REDIS_EXPOSED_PORT=.*,AI_REDIS_EXPOSED_PORT=${AI_REDIS_EXPOSED_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_ES_EXPOSED_PORT=.*,AI_ES_EXPOSED_PORT=${AI_ES_EXPOSED_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_KIBANA_EXPOSED_PORT=.*,AI_KIBANA_EXPOSED_PORT=${AI_KIBANA_EXPOSED_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^CRON_AI_EXPOSED_HTTP_PORT=.*,CRON_AI_EXPOSED_HTTP_PORT=${CRON_AI_EXPOSED_HTTP_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^CRON_AI_EXPOSED_GRPC_PORT=.*,CRON_AI_EXPOSED_GRPC_PORT=${CRON_AI_EXPOSED_GRPC_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^SCP_AI_EXPOSED_PORT=.*,SCP_AI_EXPOSED_PORT=${SCP_AI_EXPOSED_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^UI_AI_EXPOSED_PORT=.*,UI_AI_EXPOSED_PORT=${UI_AI_EXPOSED_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^TRAINING_EXPOSED_PORT=.*,TRAINING_EXPOSED_PORT=${TRAINING_EXPOSED_PORT}," "$PracticalEnv"
# 处理启动端口：非空则将environment.env文件变量替换给.env文件指定变量
$ReplaceCommand  "s,^AI_REDIS_START_PORT=.*,AI_REDIS_START_PORT=${AI_REDIS_START_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_ES_START_PORT=.*,AI_ES_START_PORT=${AI_ES_START_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_KIBANA_START_PORT=.*,AI_KIBANA_START_PORT=${AI_KIBANA_START_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_CONSUL_HTTP_START_PORT=.*,AI_CONSUL_HTTP_START_PORT=${AI_CONSUL_HTTP_START_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_CONSUL_HTTP2_START_PORT=.*,AI_CONSUL_HTTP2_START_PORT=${AI_CONSUL_HTTP2_START_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^CRON_AI_START_HTTP_PORT=.*,CRON_AI_START_HTTP_PORT=${CRON_AI_START_HTTP_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^CRON_AI_START_GRPC_PORT=.*,CRON_AI_START_GRPC_PORT=${CRON_AI_START_GRPC_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^SCP_AI_START_PORT=.*,SCP_AI_START_PORT=${SCP_AI_START_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^UI_AI_START_PORT=.*,UI_AI_START_PORT=${UI_AI_START_PORT}," "$PracticalEnv"
$ReplaceCommand  "s,^TRAINING_START_PORT=.*,TRAINING_START_PORT=${TRAINING_START_PORT}," "$PracticalEnv"
# 处理容器名：非空则将environment.env文件变量替换给.env文件指定变量
$ReplaceCommand  "s,^AI_CONSUL_HOST=.*,AI_CONSUL_HOST=${AI_CONSUL_HOST}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_REDIS_HOST=.*,AI_REDIS_HOST=${AI_REDIS_HOST}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_ES_HOST=.*,AI_ES_HOST=${AI_ES_HOST}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_KIBANA_HOST=.*,AI_KIBANA_HOST=${AI_KIBANA_HOST}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_CRON_HOST=.*,AI_CRON_HOST=${AI_CRON_HOST}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_FB_HOST=.*,AI_FB_HOST=${AI_FB_HOST}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_SCP_HOST=.*,AI_SCP_HOST=${AI_SCP_HOST}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_UI_HOST=.*,AI_UI_HOST=${AI_UI_HOST}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_UI_HOST=.*,AI_UI_HOST=${AI_UI_HOST}," "$PracticalEnv"
$ReplaceCommand  "s,^AI_TRAINING_HOST=.*,AI_TRAINING_HOST=${AI_TRAINING_HOST}," "$PracticalEnv"
# 处理网络名称：非空则将environment.env文件变量替换给.env文件指定变量
$ReplaceCommand  "s,^DOCKER_NETWORK_NAME=.*,DOCKER_NETWORK_NAME=${DOCKER_NETWORK_NAME}," "$PracticalEnv"


## ENV=./environment/local.env
# 处理服务之间环境变量：非空则将environment.env文件变量替换给environment.env文件指定变量
if [ -z $SCP_GO_AI_REDIS_HOST ]; then
 $ReplaceCommand  "s,^SCP_GO_AI_REDIS_HOST=.*,SCP_GO_AI_REDIS_HOST=${AI_REDIS_HOST}," "$ENV"
fi
if [ -z $SCP_GO_AI_REDIS_HOST ]; then
 $ReplaceCommand  "s,^SCP_GO_AI_REDIS_PORT=.*,SCP_GO_AI_REDIS_PORT=${AI_REDIS_START_PORT}," "$ENV"
fi
#if [ -z $SCP_GO_AI_REDIS_PASSWORD ]; then
 #$ReplaceCommand  "s,^SCP_GO_AI_REDIS_PASSWORD=.*,SCP_GO_AI_REDIS_PASSWORD=${}," "$ENV"
#fi
if [ -z $SCP_GO_AI_CRON_HOST ]; then
 $ReplaceCommand  "s,^SCP_GO_AI_CRON_HOST=.*,SCP_GO_AI_CRON_HOST=${AI_CRON_HOST}," "$ENV"
fi
if [ -z $SCP_GO_AI_CRON_PORT ]; then
 $ReplaceCommand  "s,^SCP_GO_AI_CRON_PORT=.*,SCP_GO_AI_CRON_PORT=${CRON_AI_START_HTTP_PORT}," "$ENV"
fi
if [ -z $SCP_GO_AI_ES_HOST ]; then
 $ReplaceCommand  "s,^SCP_GO_AI_ES_HOST=.*,SCP_GO_AI_ES_HOST=${AI_ES_HOST}," "$ENV"
fi
if [ -z $SCP_GO_AI_ES_PORT ]; then
 $ReplaceCommand  "s,^SCP_GO_AI_ES_PORT=.*,SCP_GO_AI_ES_PORT=${AI_ES_START_PORT}," "$ENV"
fi
if [ -z $SCP_GO_AI_TRAINING_HOST ]; then
 $ReplaceCommand  "s,^SCP_GO_AI_TRAINING_HOST=.*,SCP_GO_AI_TRAINING_HOST=${AI_TRAINING_HOST}," "$ENV"
fi
if [ -z $SCP_GO_AI_TRAINING_PORT ]; then
 $ReplaceCommand  "s,^SCP_GO_AI_TRAINING_PORT=.*,SCP_GO_AI_TRAINING_PORT=${TRAINING_START_PORT}," "$ENV"
fi
if [ -z $CRON_AI_CONSUL_PORT ]; then
 $ReplaceCommand  "s,^CRON_AI_CONSUL_PORT=.*,CRON_AI_CONSUL_PORT=${AI_CONSUL_HTTP_START_PORT}," "$ENV"
fi
if [ -z $CRON_AI_CONSUL_HOST ]; then
 $ReplaceCommand  "s,^CRON_AI_CONSUL_HOST=.*,CRON_AI_CONSUL_HOST=${AI_CONSUL_HOST}," "$ENV"
fi
if [ -z $CRON_AI_REDIS_HOST ]; then
 $ReplaceCommand  "s,^CRON_AI_REDIS_HOST=.*,CRON_AI_REDIS_HOST=${AI_REDIS_HOST}," "$ENV"
fi
if [ -z $CRON_AI_REDIS_PORT ]; then
 $ReplaceCommand  "s,^CRON_AI_REDIS_PORT=.*,CRON_AI_REDIS_PORT=${AI_REDIS_START_PORT}," "$ENV"
fi
#if [ -z $CRON_AI_REDIS_PASSWORD ]; then
 #$ReplaceCommand  "s,^CRON_AI_REDIS_PASSWORD=.*,CRON_AI_REDIS_PASSWORD=${}," "$ENV"
#fi
if [ -z $CRON_AI_SCP_GO_AI_HOST ]; then
 $ReplaceCommand  "s,^CRON_AI_SCP_GO_AI_HOST=.*,CRON_AI_SCP_GO_AI_HOST=${AI_SCP_HOST}," "$ENV"
fi
if [ -z $CRON_AI_SCP_GO_AI_PORT ]; then
 $ReplaceCommand  "s,^CRON_AI_SCP_GO_AI_PORT=.*,CRON_AI_SCP_GO_AI_PORT=${SCP_AI_START_PORT}," "$ENV"
fi
if [ -z $UI_SCP_GO_AI_HOST ]; then
 $ReplaceCommand  "s,^UI_SCP_GO_AI_HOST=.*,UI_SCP_GO_AI_HOST=${AI_SCP_HOST}," "$ENV"
fi
if [ -z $UI_SCP_GO_AI_PORT ]; then
 $ReplaceCommand  "s,^UI_SCP_GO_AI_PORT=.*,UI_SCP_GO_AI_PORT=${SCP_AI_START_PORT}," "$ENV"
fi



$ReplaceCommand "s,^ENV=.*,ENV=${EE}," "$PracticalEnv"