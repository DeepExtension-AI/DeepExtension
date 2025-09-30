"""
 /*
  * Copyright 2025 DeepExtension team
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  *     http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */
"""
import os
import hashlib
import requests
import traceback
import json
from train_callback import write_log
from enums import LogEnum,LevelEnum,StatusEnum
class OllamaUploader:
    def __init__(self, server_url,train_id,seq):
        """Initialize uploader and record server address"""
        self.server_url = server_url
        self.train_id=train_id
        self.seq=seq
        write_log(LevelEnum.INFO,LogEnum.DeployInitializeOllama,server_url,train_id,seq,None)

    def calculate_sha256(self, file_path):
        """Calculate file SHA256 hash value"""
        write_log(LevelEnum.INFO,LogEnum.DeployCalculateSha256Start,file_path,self.train_id,self.seq,None)
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            digest = sha256.hexdigest()
            write_log(LevelEnum.INFO,LogEnum.DeployCalculateSha256Success,file_path+","+digest,self.train_id,self.seq,None)
            return digest
        except Exception as e:
            write_log(LevelEnum.ERROR,LogEnum.DeployCalculateSha256Failed,file_path+","+str(e),self.train_id,self.seq,None)
            raise

    def upload_file(self, file_path):
        """Upload single file and return SHA256 checksum"""
        try:
            sha256_digest = self.calculate_sha256(file_path)
            upload_url = f"{self.server_url}/api/blobs/sha256:{sha256_digest}"
            write_log(LevelEnum.INFO,LogEnum.DeployUploadingFiles,file_path+","+upload_url,self.train_id,self.seq,None)

            with open(file_path, "rb") as f:
                response = requests.post(upload_url, data=f, timeout=30)

            if response.status_code == 201:
                write_log(LevelEnum.INFO,LogEnum.DeployUploadFileSuccess,file_path,self.train_id,self.seq,None)

                return sha256_digest
            elif response.status_code == 200:
                write_log(LevelEnum.INFO,LogEnum.DeployUploadFileSkip,file_path,self.train_id,self.seq,None)
                return sha256_digest
            else:
                write_log(LevelEnum.ERROR,LogEnum.DeployUploadFileFailed,f"{file_path},{response.status_code},{response.text[:500]}",self.train_id,self.seq,None)
                return None

        except Exception as e:
            write_log(LevelEnum.ERROR,LogEnum.DeployUploadFileError,f"{file_path},{str(e)}",self.train_id,self.seq,None)
            return None

    def upload_folder(self, folder_path):
        """Batch upload files from folder"""
        write_log(LevelEnum.INFO,LogEnum.DeployUploadingFolders,f"{folder_path}",self.train_id,self.seq,None)
        success_count = 0
        failed_files = []

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                write_log(LevelEnum.INFO,LogEnum.DeployUploadFoldersStart,f"{folder_path}",self.train_id,self.seq,None)
                result = self.upload_file(file_path)
                if result:
                    success_count += 1
                else:
                    failed_files.append(file_name)
        write_log(LevelEnum.INFO,LogEnum.DeployUploadFoldersSuccess,f"{folder_path},{success_count}",self.train_id,self.seq,None)
        if failed_files:
            write_log(LevelEnum.ERROR,LogEnum.DeployUploadFoldersFailed,f"{','.join(failed_files)}",self.train_id,self.seq,None)
        return success_count == 0 

    def get_files_sha256(self, folder_path):
        """Get SHA256 mapping of all files in the folder"""
        write_log(LevelEnum.INFO,LogEnum.DeployGenerateHash,f"{folder_path}",self.train_id,self.seq,None)
        sha256_map = {}
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                try:
                    sha256 = self.calculate_sha256(file_path)
                    sha256_map[file_name] = f"sha256:{sha256}"
                    write_log(LevelEnum.INFO,LogEnum.DeployGenerateHashSuccess,f"{file_name},{sha256[:8]}",self.train_id,self.seq,None)
                except Exception as e:
                    write_log(LevelEnum.WARNING,LogEnum.DeployGenerateHashSkip,f"{file_name},{str(e)}",self.train_id,self.seq,None)
                    
        write_log(LevelEnum.INFO,LogEnum.DeployHandleWithHash,f"{len(sha256_map)}",self.train_id,self.seq,None)
                    
        return sha256_map

    def create_model(
        self,
        folder_path,
        model_name,
        quantize=None,
        template="default",
        system="You are a helpful assistant.",
        parameters=None
    ):
        """Create model and send configuration to Ollama server"""
        write_log(LevelEnum.INFO,LogEnum.DeployCreateModelStart,f"{model_name}",self.train_id,self.seq,None)

        file_map = self.get_files_sha256(folder_path)
        
        create_url = f"{self.server_url}/api/create"
        write_log(LevelEnum.INFO,LogEnum.DeployCreateModelRequest,f"{create_url}",self.train_id,self.seq,None)

        payload = {
            "model": model_name,
            "files": file_map,
            "template": template,
            "system": system,
            "parameters": parameters or {
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }

        if quantize and str(quantize).upper() != "NONE":
            payload["quantize"] = quantize
            write_log(LevelEnum.INFO,LogEnum.DeployEnableQuantize,f"{quantize}",self.train_id,self.seq,None)
        
        try:
            write_log(LevelEnum.INFO,LogEnum.DeploySendingCreateModel,None,self.train_id,self.seq,None)

            response = requests.post(create_url, json=payload, timeout=None)
            write_log(LevelEnum.INFO,LogEnum.DeployCreateStatusCode,f"{response.status_code}",self.train_id,self.seq,None)

            if response.status_code == 200:
                write_log(LevelEnum.INFO,LogEnum.DeployCreateModelSuccess,f"{model_name}",self.train_id,self.seq,None)
            else:
                write_log(LevelEnum.ERROR,LogEnum.DeployCreateModelFailed,f"{response.status_code},{response.text[:500]}",self.train_id,self.seq,None)
            for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line.decode('utf-8'))
                            if "error" in json_response:
                                write_log(LevelEnum.WARNING,LogEnum.DeployCreateModelRawResponse,f"{json_response}",self.train_id,self.seq,None)
                                return False
                        except json.JSONDecodeError as e:
                            write_log(LevelEnum.WARNING,LogEnum.DeployCreateModelRawResponse,f"{line}",self.train_id,self.seq,None)

            return response.status_code == 200

        except requests.exceptions.RequestException as e:
            write_log(LevelEnum.ERROR,LogEnum.DeployNetworkError,f"{str(e)}",self.train_id,self.seq,None)    
            return False

# Model quantization method (optional)
# Available quantization methods: q4_K_M, q4_K_S, q8_0 (default: no quantization)
QUANTIZE = None

# Custom prompt template and system instructions (optional)
TEMPLATE = """{{- if .System }}{{ .System }}{{ end }}\n{{- range $i, $_ := .Messages }}\n{{- $last := eq (len (slice $.Messages $i)) 1}}\n{{- if eq .Role "user" }}<｜User｜>{{ .Content }}\n{{- else if eq .Role "assistant" }}<｜Assistant｜>{{ .Content }}{{- if not $last }}<｜end▁of▁sentence｜>{{- end }}\n{{- end }}\n{{- if and $last (ne .Role "assistant") }}<｜Assistant｜>{{- end }}\n{{- end }}"""

# Exposed to users, can be overridden
SYSTEM_PROMPT = "You are a precise AI assistant. Please provide clear, step-by-step answers."

# Custom parameters (optional)
PARAMETERS = {
    "temperature": 0.5,
    "top_p": 0.95,
    "repeat_penalty": 1.15
}

from train_callback import write_log,StatusEnum
class ModelDeployerOllama:
    @classmethod
    def deploy_model(cls, train_id: str, seq: int, status_file: str, folder_path: str, server_url: str, model_name: str,
                    quantize: str = QUANTIZE, template: str = TEMPLATE, system: str = SYSTEM_PROMPT, parameters: dict = PARAMETERS) -> dict:
        """Core business logic for model deployment

        Args:
            train_id: Task ID
            seq: Default 1
            folder_path: Path to save the model after deployment
            server_url: Ollama server URL
            model_name: Name of the deployed model
            quantize: Quantization methods available: 'q4_K_M', 'q4_K_S', 'q8_0', or None (default, no quantization)
            template: Custom template and system instructions
            system: System prompt
            parameters: Custom parameters

        Returns:
            A dictionary containing the operation results
        """
        try:
            import platform
            from urllib.parse import urlparse, urlunparse
            
            def replace_docker_host_in_url(url):
                if platform.system() == 'Darwin':  
                    parsed_url = urlparse(url)
                    if 'host.docker.internal' in parsed_url.netloc:
                        new_netloc = parsed_url.netloc.replace('host.docker.internal', '127.0.0.1')
                        new_parsed_url = parsed_url._replace(netloc=new_netloc)
                        return urlunparse(new_parsed_url)
                return url
            def check_web_service(url: str, timeout=3) -> bool:

                try:
                    url=replace_docker_host_in_url(url)
                    print(url)
                    r = requests.get(url, timeout=timeout)
                    return r.status_code < 400  
                except requests.exceptions.RequestException as e:
                    return False

            if not check_web_service(server_url):
                write_log(LevelEnum.ERROR, LogEnum.DeployOllamaError, f"{server_url}", train_id, seq, None) 
                return {
                    "status": "error",
                    "message": "Failed to connect to Ollama service. "
                }
            server_url=replace_docker_host_in_url(server_url)
            uploader = OllamaUploader(server_url,train_id,seq)

            has_failed = uploader.upload_folder(folder_path)

            success = uploader.create_model(
                folder_path,
                model_name,
                template=template or TEMPLATE,
                system=system,
                parameters=parameters,
                quantize=quantize
            )
            if success:
                return {
                    "status": "success",
                    "message": "Model deployed successfully"
                }
            else:
                write_log(LevelEnum.ERROR, LogEnum.DeployNotFullSuccess, None, train_id, seq, None)    
                return {
                    "status": "error",
                    "message": "Partial success: files uploaded but model creation failed"
                }

        except Exception as e:
            write_log(LevelEnum.ERROR,LogEnum.DeployError,f"{str(e)},{type(e).__name__},{traceback.format_exc()}",train_id,seq,None)    
            return {
                "status": "error",
                "message": str(e)
            }

# 在文件末尾添加以下代码
if __name__ == "__main__":
    import sys
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Deploy Ollama ')
    
    # 定义命令行参数
    parser.add_argument('--trainId', type=str, required=True, help='训练ID')
    parser.add_argument('--seq', type=str, default='1', help='序列号')
    parser.add_argument('--folderPath', type=str, required=True, help='文件夹路径')
    parser.add_argument('--serverUrl', type=str, required=True, help='服务器URL')
    parser.add_argument('--modelName', type=str, required=True, help='模型名称')
    parser.add_argument('--quantize', type=str, default='', help='量化方式')
    parser.add_argument('--template', type=str, default='', help='模板')
    parser.add_argument('--system', type=str, default='',  help='系统提示')
    parser.add_argument('--parameters', type=str, default='', help='参数字典')
    
    known_args, unknown_args = parser.parse_known_args()
    print("Known args:", known_args)
    print("Unknown args:", unknown_args)
    args = known_args
    
    try:
        
        # 执行部署
        result = ModelDeployerOllama.deploy_model(
    
            status_file='status_file',
            train_id = args.trainId,
            seq=args.seq,
            folder_path = args.folderPath,
            model_name = args.modelName,
            server_url = args.serverUrl,
            quantize = args.quantize,
            template = args.template,
            system = args.system,
            parameters = args.parameters,
        )
        
        # 输出结果
        print(json.dumps(result))
        
        # 根据结果设置退出码
        if result.get('status') == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Script execution failed: {str(e)}"
        }
        print(json.dumps(error_result))
        sys.exit(1)