# 运行准备脚本
SYSTEM=$(uname -s)
source ./.env
echo "Environment: $GLOBAL_ENV"
FileLocation="./"
PROJECT_NAME=scp-ai-$(echo $GLOBAL_ENV | cut -d '.' -f 1)
if [ "$SYSTEM" = "Darwin" ];then
  ReplaceCommand="sed_-i_\'\'"
  ./set_env_variable.sh $ReplaceCommand $FileLocation
  docker-compose -p ${PROJECT_NAME} -f ./after-set-variable-mac.yml --env-file ./.env up -d --remove-orphans
  APP_NAME="training-py"
  APP_SCRIPT=$PYTHON_CODE_PATH
  ## 需要提前安装pm2: npm install pm2 -g
  ## 查看Python容器日志 pm2 logs training-py
  # 检查服务是否存在
  cd deep-e-python
  if pm2 list | grep -q $APP_NAME; then
      echo "🔄 Restarting $APP_NAME..."
      pm2 restart $APP_NAME
  else
      echo "🚀 Starting $APP_NAME..."
      pm2 start $APP_SCRIPT --name $APP_NAME
  fi
  pm2 save
else if [ "$SYSTEM" = "Linux" ];then
    ReplaceCommand='sed_-i'
    ./set_env_variable.sh $ReplaceCommand $FileLocation
     docker-compose -p ${PROJECT_NAME} -f ./after-set-variable.yml --env-file ./.env up  -d --remove-orphans
    fi
fi

