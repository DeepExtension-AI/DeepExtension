# 运行准备脚本
SYSTEM=$(uname -s)
source ./.env
echo "Environment: $GLOBAL_ENV"
FileLocation="./"
PROJECT_NAME=scp-ai-$(echo $GLOBAL_ENV | cut -d '.' -f 1)
if [ "$SYSTEM" = "Darwin" ];then
  ReplaceCommand="sed_-i_\'\'"
  ./set_env_variable.sh $ReplaceCommand $FileLocation
  docker-compose -p ${PROJECT_NAME} -f ./after-set-variable.yml --env-file ./.env up -d --remove-orphans
else if [ "$SYSTEM" = "Linux" ];then
    ReplaceCommand='sed_-i'
    ./set_env_variable.sh $ReplaceCommand $FileLocation
     docker-compose -p ${PROJECT_NAME} -f ./after-set-variable.yml --env-file ./.env up -d --remove-orphans
    fi
fi