import subprocess

import os
import sys

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import hashlib
import requests
import traceback
import json
from train_callback import write_log
from enums import LogEnum, LevelEnum, StatusEnum

if __name__ == "__main__":
    import sys
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Deploy Ollama ')
    
    # 定义命令行参数
    parser.add_argument('--trainId', type=str, required=True, help='训练ID')
    parser.add_argument('--seq', type=str, default='1', help='序列号')
   # parser.add_argument('--folderPath', type=str, required=True, help='文件夹路径')
    parser.add_argument('--serverUrl', type=str, required=True, help='服务器URL')
    parser.add_argument('--modelName', type=str, required=True, help='模型名称')
    #parser.add_argument('--quantize', type=str, default='', help='量化方式')
    #parser.add_argument('--template', type=str, default='', help='模板')
    parser.add_argument('--system', type=str, default='',  help='系统提示')
    #parser.add_argument('--parameters', type=str, default='', help='参数字典')
    parser.add_argument('--basePath', type=str, default='',  help='基础模型')
    parser.add_argument('--actualPath', type=str, default='', help='lora模型')

    known_args, unknown_args = parser.parse_known_args()
    print("Known args:", known_args)
    print("Unknown args:", unknown_args)
    args = known_args
    
    try:
        # 构建请求参数
        deepESdParam = {
            "model_id": args.modelName,
            "name": args.modelName,
            "pipeline_class": "StableDiffusion3Pipeline",
            "type": "stable_diffusion",
            "base_model_path":args.basePath,
            "lora_weights_path":args.actualPath+"/pytorch_lora_weights.safetensors",
            "lora_scale":0.7
        }
        
        
        # 发送POST请求
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            args.serverUrl+"/models",
            data=json.dumps(deepESdParam),
            headers=headers,
            timeout=300  # 设置5分钟超时
        )
        print(response)
        # 检查响应状态
        if response :
            result = {
                "status": "success",
                "message": f"模型部署成功: {args.modelName}",
                "response": response.json() if response.content else {}
            }
        else:
            result = {
                "status": "error",
                "message": f"部署请求失败，状态码: {response.status_code}",
                "response_text": response.text
            }
        
        print(json.dumps(result))
        
        # 根据结果设置退出码
        if result.get('status') == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except requests.exceptions.RequestException as e:
        error_result = {
            "status": "error",
            "message": f"网络请求错误: {str(e)}"
        }
        print(json.dumps(error_result))
    
        sys.exit(1)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"脚本执行失败: {str(e)}",
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result))
        
        sys.exit(1)