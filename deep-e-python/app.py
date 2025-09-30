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

from flask import Flask, request, jsonify,Response
from queue import Queue
import os
import traceback
from train_callback import write_log,StatusEnum
from datetime import datetime
import platform
import time
from typing import List, Dict, Optional, Union

from train_callback import write_log
from enums import LogEnum,LevelEnum,StatusEnum
import threading
import subprocess
from redis_util import redis_client
import json
from http import HTTPStatus



app = Flask(__name__)

@app.route('/health')
def health_check():  
    return jsonify({"status": HTTPStatus.OK.phrase}), HTTPStatus.OK.value


from run_train import Training

@app.route('/transferTrain', methods=['POST'])
def transferTrain():
    '''
    Endpoint for initiating transfer learning training
    
    Receives training parameters and starts the training process in a separate thread
    '''
    data = request.json
    train_id = data.get("train_id")
    seq = data.get("seq")
    
    # Log the incoming request parameters
    write_log(LevelEnum.INFO, LogEnum.ActualParams, f"{request.json}", train_id, seq, None)
    
    # Validate required parameters
    required_params = ['train_id', 'seq']
    if not all(param in data for param in required_params):
        write_log(LevelEnum.ERROR, LogEnum.ParamCheckFailed, ", ".join(required_params), train_id, seq, None)
        return jsonify(
            {
                'status': HTTPStatus.BAD_REQUEST.phrase,
                'train_id': train_id,
                'error': 'Missing required parameters'
            }
        ), HTTPStatus.BAD_REQUEST.value
    
    try:
        # Start training in a separate thread
        threading.Thread(
            target=Training.run_training,
            args=(train_id, data)
        ).start()
        
        return jsonify(
            {
                'status': HTTPStatus.OK.phrase,
                'train_id': train_id,
                'error': ''
            }
        ), HTTPStatus.OK.value
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
        write_log(LevelEnum.ERROR, LogEnum.TrainingFailed, None, str(e), train_id, seq, None)
        return jsonify(
            {
                'status': HTTPStatus.BAD_REQUEST.phrase,
                'train_id': train_id,
                'error': error_msg
            }
        ), HTTPStatus.BAD_REQUEST.value

def get_absolute_path(relative_path: Optional[Union[str, None]]) -> Optional[str]:
    """
    Convert a relative path to an absolute path, with special case handling
    
    Args:
        relative_path: The input path (may be None, empty string, or "None")
        
    Returns:
        str: The converted absolute path if valid
        None: If the path is None, empty string, or "None"
        
    Notes:
        - Returns None if the path doesn't exist
        - Handles string "None" as a special case (returns None)
    """
    # Handle special cases
    if relative_path is None or relative_path == "" or relative_path == "None":
        return None
        
    # Normal path conversion logic
    absolute_path = os.path.abspath(relative_path)
    
    # Return None if path doesn't exist (alternative: could raise exception)
    if not os.path.exists(absolute_path):
        return None
        
    return absolute_path

@app.route('/chatTest', methods=['POST'])
def chatWithTest():
    # Get all required data early in request context
    try:
        data = request.get_json()
        if not data or 'modelName' not in data or 'prompt' not in data:
            return jsonify({"error": "Missing required parameters: modelName and prompt", "status": "error"}), 400
    except Exception as e:
        error_msg = f"Request parsing failed: {str(e)}\n{traceback.format_exc()}"
        return jsonify({"status": "error", "error": error_msg}), 400

    def linux_generate(data):
        """Stream generated responses"""
        try:
            result_queue = Queue()
            
            def run_chat_process(data):
                """Run model inference in subprocess"""
                try:
                    # Build command line arguments
                    chat_args = [
                        f"--promptFile={data.get('promptFile', '')}",
                        f"--condaEnv={data.get('condaEnv', '')}",
                        f"--model_id={data.get('modelId', '')}",
                        f"--model_name={data['modelName']}",
                        f"--base_model_path={data['baseModelPath']}",
                        f"--prompt={data['prompt']}",
                        f"--session_id={data.get('sessionId', 'default')}",
                        f"--max_length={data.get('maxLength', 512)}",
                        f"--max_context_tokens={data.get('maxContentTokens', 2048)}",
                        f"--temperature={data.get('temperature', 0.7)}",
                        f"--top_p={data.get('topP', 0.9)}",
                        f"--top_k={data.get('topK', 50)}",
                        f"--model_type={data.get('modelType', 'chat')}",
                        f"--template={data.get('template','')}"
                    ]

                    # Add optional parameters
                    if 'systemPrompt' in data:
                        chat_args.append(f"--system_prompt={data['systemPrompt']}")
                    if 'resetContext' in data:
                        chat_args.append(f"--reset_context={str(data['resetContext']).lower()}")
                    if 'history' in data:
                        chat_args.append(f"--history={json.dumps(data['history'])}")

                    system = platform.system()
                    if system == "Linux":
                        cmd = ['python3', 'linux_code/model_manager.py'] + chat_args
                    else:
                        print("Unknown OS") 
                        
                    print(f"Executing command: {' '.join(cmd)}")

                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        start_new_session=True
                    )

                    # Process real-time output
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            output = output.strip()
                            print(f"print:{output}")
                            try:
                                # Try parsing as JSON, treat as plain output if not JSON
                                outputDic = json.loads(output)
                                if outputDic["status"] == "resp":
                                    tmp = json.dumps({"text": outputDic["text"]})
                                    print(f"yield:{tmp}")
                                    yield f"data: {tmp}\n\n"
                                elif outputDic["status"] == "error":
                                    tmp = json.dumps({"finish_reason": outputDic["error"]})
                                    print(f"yield_error:{tmp}")
                                    yield f"data: {tmp}\n\n"
                            except:
                                print(f"print——except:{output}")
                                if 'status' in output:
                                    tmp = json.dumps({"text": output})
                                    print(f"except:{tmp}")
                                    yield f"data: {tmp}\n\n"

                    # Get final result
                    return_code = process.poll()
                    stdout, stderr = process.communicate()

                    if return_code == 0:
                        try:
                            result = json.loads(stdout.splitlines()[-1])
                            if result.get("status") == "resp":  
                                tmp = json.dumps({"finish_reason": None, "text": result.get("text")})
                                #print(f"stderr1:{tmp}")
                                ##yield f"data: {tmp}\n\n"
                        except:
                            tmp = json.dumps({"finish_reason": stderr})
                            #print(f"stderr2:{tmp}")
                            ##yield f"data: {tmp}\n\n"
                    else:
                        
                        tmp = json.dumps({"finish_reason": stderr, "text": ""})
                        #print(f"stderr3:{tmp}")
                        ##yield f"data: {tmp}\n\n"

                except Exception as e:
                    error_msg = f"Chat process execution failed: {str(e)}\n{traceback.format_exc()}"
                    tmp = json.dumps({"finish_reason": error_msg, "text": ""}) 
                    yield f"data: {tmp}\n\n"

            # Start processing thread
            chat_thread = threading.Thread(
                target=lambda q, d: [q.put(line) for line in run_chat_process(d)],
                args=(result_queue, data)
            )
            chat_thread.start()

            # Get output from queue and forward it
            while chat_thread.is_alive() or not result_queue.empty():
                try:
                    line = result_queue.get(timeout=2)
                    tmp = line
                    print(
                        f"alive:{tmp}"
                    )
                    yield f"{tmp}\n\n"
                except Exception as e:
                    yield "\n"  
                    continue

        except Exception as e:
            error_msg = f"Response generation failed: {str(e)}\n{traceback.format_exc()}"
            tmp = json.dumps({"finish_reason": error_msg, "text": ""})
            yield f"data: {tmp}\n\n"

    def mac_generate(data):
        try:
            model_id = data["modelId"]
            base_model_path = data["baseModelPath"]
            history = data.get("history", [])
            max_content_tokens = data.get("maxContentTokens", 2048)

            absolute_base_path = get_absolute_path(base_model_path)
            absolute_adapter_path = get_absolute_path(model_id)

            # Initialize model inferencer
            from mac_code.mac_chat import ModelInference
            inferencer = ModelInference(absolute_base_path, adapter_path=absolute_adapter_path)

            # Stream generated responses
            for response in inferencer.generate_response(
                messages=history,
                max_tokens=max_content_tokens
            ):
                yield f"data: {json.dumps({'text': response})}\n\n"

        except Exception as e:
            error = f"Mac generation failed: {str(e)}\n{traceback.format_exc()}"
            yield f"data: {json.dumps({'finish_reason': error})}\n\n"

    system = platform.system()
    def generate(data):
        try:
            if system == "Linux":
                for chunk in linux_generate(data):
                    yield chunk
            
            elif system == "Darwin":
                for chunk in mac_generate(data):
                    yield chunk
            
            else:
                yield f"data: {json.dumps({'finish_reason': 'Unsupported OS'})}\n\n"
        
        except Exception as e:
                error = f"Generate Failed: {str(e)}\n{traceback.format_exc()}"
                yield f"data: {json.dumps({'finish_reason': error})}\n\n"

    return Response(
        generate(data),  
        mimetype='application/x-ndjson',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/chat', methods=['POST'])
def chatWithModel():
    # Get all required data early in request context
    try:
        data = request.get_json()
        print(data)
        if not data or 'modelName' not in data or 'prompt' not in data:
            return jsonify({"error": "Missing required parameters: modelName and prompt", "status": "error"}), 400
    except Exception as e:
        error_msg = f"Request parsing failed: {str(e)}\n{traceback.format_exc()}"
        return jsonify({"status": "error", "error": error_msg}), 400

    def linux_generate(data):
        """Stream generated responses"""
        try:
            result_queue = Queue()
            prompter = Prompting()
            chat_thread = threading.Thread(
                target=lambda q, d: [q.put(line) for line in prompter.run_prompting(d)],
                args=(result_queue, data)
            )
            chat_thread.start()

            # Get output from queue and forward it
            while chat_thread.is_alive() or not result_queue.empty():
                try:
                    line = result_queue.get(timeout=2)
                    tmp = line
                    print(
                        f"alive:{tmp}"
                    )
                    yield f"{tmp}\n\n"
                except Exception as e:
                    yield "\n"  
                    continue

        except Exception as e:
            error_msg = f"Response generation failed: {str(e)}\n{traceback.format_exc()}"
            tmp = json.dumps({"finish_reason": error_msg, "text": ""})
            yield f"data: {tmp}\n\n"

    def mac_generate(data):
        try:
            model_id = data["modelId"]
            base_model_path = data["baseModelPath"]
            history = data.get("history", [])
            max_content_tokens = data.get("maxContentTokens", 2048)

            absolute_base_path = get_absolute_path(base_model_path)
            absolute_adapter_path = get_absolute_path(model_id)

            # Initialize model inferencer
            from mac_code.mac_chat import ModelInference
            inferencer = ModelInference(absolute_base_path, adapter_path=absolute_adapter_path)

            # Stream generated responses
            for response in inferencer.generate_response(
                messages=history,
                max_tokens=max_content_tokens
            ):
                yield f"data: {json.dumps({'text': response})}\n\n"

        except Exception as e:
            error = f"Mac generation failed: {str(e)}\n{traceback.format_exc()}"
            yield f"data: {json.dumps({'finish_reason': error})}\n\n"

    system = platform.system()
    def generate(data):
        try:
            if system == "Linux":
                for chunk in linux_generate(data):
                    yield chunk
            
            elif system == "Darwin":
                for chunk in mac_generate(data):
                    yield chunk
            
            else:
                yield f"data: {json.dumps({'finish_reason': 'Unsupported OS'})}\n\n"
        
        except Exception as e:
                error = f"Generate Failed: {str(e)}\n{traceback.format_exc()}"
                yield f"data: {json.dumps({'finish_reason': error})}\n\n"

    return Response(
        generate(data),  
        mimetype='application/x-ndjson',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )
    
@app.route('/trainingStatus/<taskUuid>', methods=['GET'])
def check_training_status(taskUuid):
    status = redis_client.get_status(taskUuid)
    return jsonify(status), 200

@app.route('/mergeModel', methods=['POST'])
def merge_model():
    """
    Model merging API (automatically calls Linux/Mac implementation based on system)
    
    Example request body:
    {
        "baseModelPath": "/path/to/base_model",
        "loraAdapterPath": "/path/to/lora_adapter",
        "savePath": "/path/to/save",
        "dtypeStr": "bfloat16",
        "trainId": "xxx",
        "seq": 0
    }
    """
    data = request.get_json()
    train_id = data.get("trainId")
    seq = data.get("seq")
    try:
        write_log(LevelEnum.INFO, LogEnum.StartHandlingEvents, 'Save', train_id, seq, None)
        required_params = ['baseModelPath', 'loraAdapterPath', 'savePath', 'modelUsageType','trainId', 'seq']
        if not all(param in data for param in required_params):
            write_log(LevelEnum.ERROR, LogEnum.ParamCheckFailed, ", ".join(required_params), train_id, seq, None)
            return jsonify(
                {
                    'status': HTTPStatus.BAD_REQUEST.phrase,
                    'train_id': train_id,
                    'error': 'Missing required parameters'
                }
            ), HTTPStatus.BAD_REQUEST.value
        
        redis_client.set_status(train_id, StatusEnum.Running.value)
        base_path = data['baseModelPath']
        adapter_path = data['loraAdapterPath']
        save_path = data['savePath']
        dtype_str = data.get('dtypeStr', 'bfloat16')
        train_id = data['trainId']
        seq = data['seq']
        modelUsageType=data['modelUsageType']
        
        # Log actual parameters
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"base_path:{base_path}", train_id, seq, None)
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"adapter_path:{adapter_path}", train_id, seq, None)
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"save_path:{save_path}", train_id, seq, None)
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"dtype_str:{dtype_str}", train_id, seq, None)
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"train_id:{train_id}", train_id, seq, None)
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"seq:{seq}", train_id, seq, None)
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"modelUsageType:{modelUsageType}", train_id, seq, None)
        conda_env = data.get('condaEnv', 'base') or 'base'
        saveFile = 'linux_code/save_model'
        system_type = platform.system().lower()
        if system_type == 'darwin':
            save_path = 'mac_code/mac_save'
        pythonFile = data.get('savePythonFile', saveFile) or saveFile

        cmd = [
            "conda", "run", "-n", conda_env,
            "python", pythonFile+".py",
            "--train_id", str(data.get('trainId')),
            "--seq", str(data.get('seq','1')),
            "--save_path", str(data['savePath']),
            "--base_path", str(data['baseModelPath']),
            "--adapter_path", str(data['loraAdapterPath']),
            "--dtype_str", str(data.get('dtypeStr', 'bfloat16')),
            "--model_usage_type", str(data.get('modelUsageType', 'chat')),
        ]

        # 执行命令
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        return_code = process.returncode
        # 读取执行结果
        if return_code == 0:
            try:
                result = json.loads(stdout)
            
                if result.get('status') == 'success':
                    write_log(LevelEnum.INFO, LogEnum.HandleEventsSuccess, 'Save', train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Success.value)
                    return jsonify({
                        'status': HTTPStatus.OK.phrase,
                        'train_id': train_id,
                        'error': ''
                    }), HTTPStatus.OK.value
                else:
                    write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Save' + "," + result.get('message', ''), train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Failed.value)
                    return jsonify(result), HTTPStatus.BAD_REQUEST.value
            except json.JSONDecodeError:
                if 'success' in stdout.lower():
                    write_log(LevelEnum.INFO, LogEnum.HandleEventsSuccess, 'Save', train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Success.value)
                    return jsonify({
                        'status': HTTPStatus.OK.phrase,
                        'train_id': train_id,
                        'error': ''
                    }), HTTPStatus.OK.value
                else:
                    write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, f'Save failed: Invalid JSON output: {stdout}', train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Failed.value)
                    return jsonify({
                        'status': HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                        'train_id': train_id,
                        'error': f'Invalid JSON output: {stdout}'
                    }), HTTPStatus.INTERNAL_SERVER_ERROR.value
        else:
            write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Save' + "," + stderr, train_id, seq, None)
            redis_client.set_status(train_id, StatusEnum.Failed.value)
            return jsonify({
                'status': HTTPStatus.BAD_REQUEST.phrase,
                'train_id': train_id,
                'error': stderr
            }), HTTPStatus.BAD_REQUEST.value
        
        # System detection
 

    except Exception as e:
        write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Save' + "," + f"{str(e)}\n{traceback.format_exc()}", train_id, seq, None)
        redis_client.set_status(train_id, StatusEnum.Failed.value)
        return {
            'status': HTTPStatus.BAD_REQUEST.value,
            'train_id': data['trainId'],
            'error': str(e)
        }
    
from deploy_ollama import ModelDeployerOllama
@app.route('/deployModel', methods=['POST'])
def deploy_model():
    """Model deployment API endpoint
    Example request body:
    {
        "trainId": "",
        "seq": 1,
        "folderPath": "",
        "serverUrl": "",
        "modelName": "",
        "quantize": "",
        "template": "",
        "system": "",
        "parameters": "{}"
    }
    """
    data = request.get_json()
    train_id = data.get("trainId")
    seq = data.get("seq")
    try:
        write_log(LevelEnum.INFO, LogEnum.StartHandlingEvents, 'Deploy', train_id, seq, None)
        # Parameter validation
        print(data)
        required_params = ['folderPath', 'serverUrl', 'modelName', 'trainId', 'seq']
        if not all(param in data for param in required_params):
            write_log(LevelEnum.ERROR, LogEnum.ParamCheckFailed, None, ", ".join(required_params), train_id, seq, None)
            return jsonify(
                {
                    'status': HTTPStatus.BAD_REQUEST.phrase,
                    'train_id': train_id,
                    'error': 'Missing required parameters'
                }
            ), HTTPStatus.BAD_REQUEST.value
        
        # Call business logic
        redis_client.set_status(train_id, StatusEnum.Running.value)
        conda_env = data.get('condaEnv', 'base') or 'base'
        pythonFile = data.get('deployPythonFile', 'deploy_ollama') or 'deploy_ollama'

        cmd = [
            "conda", "run", "-n", conda_env,
            "python", pythonFile+".py",
            "--trainId", str(data.get('trainId')),
            "--seq", str(data.get('seq','1')),
            "--folderPath", str(get_absolute_path(data['folderPath'])),
            "--serverUrl", str(data['serverUrl']),
            "--modelName", str(data['modelName']),
            "--quantize", str(data.get('quantize', '')),
            "--template", str(data.get('template', '')),
            "--basePath", str(data.get('basePath', '')),
            "--actualPath", str(data.get('actualPath', '')),
        ]

        # 执行命令
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        # 读取执行结果
        if return_code == 0:
            try:
                result = json.loads(stdout)
                if result.get('status') == 'success':
                    write_log(LevelEnum.INFO, LogEnum.HandleEventsSuccess, 'Deploy', train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Success.value)
                    return jsonify(result), HTTPStatus.OK.value
                else:
                    write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Deploy' + "," + result.get('message', ''), train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Failed.value)
                    return jsonify(result), HTTPStatus.BAD_REQUEST.value
            except json.JSONDecodeError:
                if 'success' in stdout.lower():
                    write_log(LevelEnum.INFO, LogEnum.HandleEventsSuccess, 'Deploy', train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Success.value)
                    return jsonify({
                        'status': HTTPStatus.OK.phrase,
                        'train_id': train_id,
                        'error': ''
                    }), HTTPStatus.OK.value
                else:
                    write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, f'Deploy failed: Invalid JSON output: {stdout}', train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Failed.value)
                    return jsonify({
                        'status': HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                        'train_id': train_id,
                        'error': f'Invalid JSON output: {stdout}'
                    }), HTTPStatus.INTERNAL_SERVER_ERROR.value
        else:
            write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, f'Deploy failed with return code {return_code}: {stderr}', train_id, seq, None)
            redis_client.set_status(train_id, StatusEnum.Failed.value)
            return jsonify({
                'status': 'error',
                'message': f'Deployment script failed with return code {return_code}',
                'stdout': stdout,
                'stderr': stderr
            }), HTTPStatus.INTERNAL_SERVER_ERROR.value
                
    except Exception as e:
        write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Deploy' + "," + f"{str(e)}\n{traceback.format_exc()}", train_id, seq, None)
        redis_client.set_status(train_id, StatusEnum.Failed.value)
        return jsonify({
            'status': HTTPStatus.BAD_REQUEST.phrase,
            'train_id': train_id,
            'error': str(e)
        }), HTTPStatus.BAD_REQUEST.value

from run_prompt import Prompting
@app.route('/imageProcessingByModel', methods=['POST'])
def image_processing_by_model():
    try:
        import uuid
        uuid_str = str(uuid.uuid4())
        data = request.get_json()
        print(data)
        prompter = Prompting()
        thread = threading.Thread(
            target=prompter.run_generate_images,
            args=(uuid_str, data)
        )
        thread.start()

        result_url = f"/getGeneratedImages/{uuid_str}"
        return jsonify({
            'code': 0,
            'message': "",
            'data': {
                'imageUrlList': [result_url]
            }
        }), 200
    except Exception as e:
        return jsonify({
            'code': -1,
            'message': str(e),
        }), 200
    
@app.route('/getGeneratedImages/<taskId>', methods=['GET'])
def getGeneratedImages(taskId):
    try:
        status_info = redis_client.get_status(taskId)
        status=status_info.get('status')
        if status == StatusEnum.Success.value :
            base_dir = "../images/"
            valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}  # 支持的图片格式
            
            if not os.path.exists(base_dir):
                return jsonify({
                    'code': -1,
                    'message': "Workdir was not found",
                    'data': {'imageUrlList': []}
                }), 404
            
            result_urls = []
            for filename in os.listdir(base_dir):
                file_path = os.path.join(base_dir, filename)
                if (os.path.isfile(file_path) and 
                    filename.startswith(taskId) and
                    os.path.splitext(filename)[1].lower() in valid_extensions):
                    result_urls.append(f"/images/{filename}")
            
            return jsonify({
                'code': 0,
                'message': f"Find {len(result_urls)} images",
                'data': {'imageUrlList': result_urls}
            }), 200
        elif status==StatusEnum.Running.value:
            return jsonify({
                'code': StatusEnum.Running.value,
                'message': "",
                'data': {}
            }), 200
        elif status ==StatusEnum.Failed.value:
            return jsonify({
                'code': -1,
                'message': "",
                'data': {}
            }), 400
        return jsonify({
                'code': StatusEnum.Unknown.value,
                'message': "",
                'data': {}
            }), 200
    except Exception as e:
        return jsonify({
            'code': -1,
            'message': f"Error: {str(e)}",
            'data': {'imageUrlList': []}
        }), 500


def create_app():
    return app

if __name__ == '__main__':
    start_port=os.getenv('TRAINING_START_PORT')
    app.run(host='0.0.0.0', port=5054
            ,threaded=True)  
