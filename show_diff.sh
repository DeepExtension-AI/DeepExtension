# 查看本机运行的镜像
images=$(docker ps --format '{{.Image}}'| sort | uniq)

for image_item in $images
do
	local_item=$(echo "$image_item" | awk -F '/' '{print $NF}' | grep -v int)
	PROJECT_NAME=$(echo "$image_item" | awk -F '/' '{print $2}' | grep -v int)
	image_image=$(echo "$local_item" | cut -d ':' -f 1)
	image_version=$(echo "$local_item" | cut -d ':' -f 2)
  if [ "$image_version" == "latest" ];
  then
    m=$(docker inspect  jianweisoft.cn/$PROJECT_NAME/$image_image | grep hash)
    extracted_string=${m#*": "\"}
    image_version_local=${extracted_string%"\""}
    image_version_local=${image_version_local%"\","}
  else
    image_version_local=$image_version
  fi
	REP_NAME=$image_image
	REFERENCE=latest
	response=$(curl -s -X 'GET' \
	"https://jianweisoft.cn/api/v2.0/projects/$PROJECT_NAME/repositories/$REP_NAME/artifacts/$REFERENCE/tags?page=1&page_size=10&with_signature=false&with_immutable_status=false" \
	-H 'accept: application/json' \
	-H 'authorization: Basic YWRtaW46SGFyYm9yMTIzNDU=')
  # 判断 是否为数组 数组表示请求成功返回了版本号
  if [[ $(jq -r 'type == "array"' <<< "$response") == "true" ]];
  then
    version=$(jq -r '.[].name' <<< "$response")
    # 判断 是否为latest
    for v in $version
    do
      if [ "$v" != "latest" ];
      then
        if [ "$v" != "$image_version" ];
        then
          echo "镜像名称:jianweisoft.cn/$PROJECT_NAME/$image_image   harbor最新版本:$v  本地版本:$image_version"
        fi
      fi
    done
  fi
done


# 检查curl命令是否成功（返回码为0表示成功）
