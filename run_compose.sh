# 运行准备脚本
> .env
cat prod.env > .env
echo "" >> .env
cat custom.conf >> .env
echo "" >> .env
cat image.env >> .env
SYSTEM=$(uname -s)
source .env
FileLocation="./"
PROJECT_NAME=scp-ai-prod
if [ "$SYSTEM" = "Darwin" ];then
  ReplaceCommand="sed_-i_\'\'"
  docker-compose -p ${PROJECT_NAME} -f ./docker-compose.yml --env-file ./.env up -d --remove-orphans
  APP_NAME="training-py"
  APP_SCRIPT="./app.py"
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
    ## 如果读到字段
      if [ "$WITH_AI_IMAGE" = "true"];then
        docker-compose -p ${PROJECT_NAME} -f ./docker-compose.yml  up  -d --remove-orphans --profile gpu
      else
         docker-compose -p ${PROJECT_NAME} -f ./docker-compose.yml  up  -d --remove-orphans
      fi
    fi
fi

