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
    train_id = data.get("taskUuid")
    seq = data.get("seq")
    
    # Log the incoming request parameters
    write_log(LevelEnum.INFO, LogEnum.ActualParams, f"{request.json}", train_id, seq, None)
    
    # Validate required parameters
    required_params = ['taskUuid', 'seq']
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

@app.route('/chat', methods=['POST'])
def chatWithModel():
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
                        f"--model_id={data.get('modelId', '')}",
                        f"--model_name={data['modelName']}",
                        f"--base_model_path={data['baseModelPath']}",
                        f"--prompt={data['prompt']}",
                        f"--session_id={data.get('sessionId', 'default')}",
                        f"--max_length={data.get('maxLength', 512)}",
                        f"--max_context_tokens={data.get('maxContextTokens', 2048)}",
                        f"--temperature={data.get('temperature', 0.7)}",
                        f"--top_p={data.get('topP', 0.9)}",
                        f"--top_k={data.get('topK', 50)}",
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
                        cmd = ['python3', 'model_manager.py'] + chat_args
                    else:
                        print("Unknown OS") 
                        
                    print(f"Executing command: {' '.join(cmd)}")

                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )

                    # Process real-time output
                    while True:
                        output = process.stdout.readline()
                        if output == '' and process.poll() is not None:
                            break
                        if output:
                            output = output.strip()
                            try:
                                # Try parsing as JSON, treat as plain output if not JSON
                                outputDic = json.loads(output)
                                if outputDic["status"] == "resp":
                                    tmp = json.dumps({"text": outputDic["text"]})
                                    yield f"data: {tmp}\n\n"
                            except:
                                if 'status' in output:
                                    tmp = json.dumps({"text": output})
                                    yield f"data: {tmp}\n\n"

                    # Get final result
                    return_code = process.poll()
                    stdout, stderr = process.communicate()

                    if return_code == 0:
                        try:
                            result = json.loads(stdout.splitlines()[-1])
                            if result.get("status") == "resp":  
                                tmp = json.dumps({"finish_reason": None, "text": result.get("text")})
                        except:
                            tmp = json.dumps({"finish_reason": stderr})
                    else:
                        tmp = json.dumps({"finish_reason": stderr, "text": ""})
                        yield f"data: {tmp}\n\n"

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
        required_params = ['baseModelPath', 'loraAdapterPath', 'savePath', 'trainId', 'seq']
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
        
        # Log actual parameters
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"base_path:{base_path}", train_id, seq, None)
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"adapter_path:{adapter_path}", train_id, seq, None)
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"save_path:{save_path}", train_id, seq, None)
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"dtype_str:{dtype_str}", train_id, seq, None)
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"train_id:{train_id}", train_id, seq, None)
        write_log(LevelEnum.INFO, LogEnum.ActualParams, f"seq:{seq}", train_id, seq, None)
        
        # System detection
        system_type = platform.system().lower()

        result = None
        if system_type == "linux":
            try:
                from linux_code.save_model import ModelMerger
                result = ModelMerger.merge_models(
                    train_id=train_id,
                    seq=seq,
                    base_model_path=base_path,
                    lora_adapter_path=adapter_path,
                    save_path=save_path,
                    dtype_str=dtype_str,
                )
                if result['status'] == 'success':
                    write_log(LevelEnum.INFO, LogEnum.HandleEventsSuccess, 'Save', train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Success.value)
                    return jsonify({
                        'status': HTTPStatus.OK.phrase,
                        'train_id': train_id,
                        'error': ''
                    }), HTTPStatus.OK.value
                else:
                    write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Save' + "," + result['message'], train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Failed.value)
                    return jsonify({
                        'status': HTTPStatus.BAD_REQUEST.phrase,
                        'train_id': train_id,
                        'error': result['message']
                    }), HTTPStatus.BAD_REQUEST.value
            except Exception as e:
                write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Save' + "," + f"Linux:{str(e)}", train_id, seq, None)
                redis_client.set_status(train_id, StatusEnum.Failed.value)
                return jsonify({
                    'status': HTTPStatus.BAD_REQUEST.phrase,
                    'train_id': train_id,
                    'error': f"Linux model merge failed: {str(e)}"
                }), HTTPStatus.BAD_REQUEST.value

        elif system_type == "darwin":  # macOS
            try:
                from mac_code.mac_save import merge_model as mac_merge_model
                result = mac_merge_model(
                    train_id=train_id,
                    seq=seq,
                    base_model_path=os.path.abspath(base_path),
                    adapter_path=os.path.abspath(adapter_path),
                    save_path=os.path.abspath(save_path),
                    dequantize_model=(dtype_str == 'float32')
                )
                if result['status'] == 'success':
                    write_log(LevelEnum.INFO, LogEnum.HandleEventsSuccess, '', train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Success.value)
                    return jsonify({
                        'status': HTTPStatus.OK.phrase,
                        'train_id': train_id,
                        'error': ''
                    }), HTTPStatus.OK.value
                else:
                    write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Save' + "," + result['message'], train_id, seq, None)
                    redis_client.set_status(train_id, StatusEnum.Failed.value)
                    return jsonify({
                        'status': HTTPStatus.BAD_REQUEST.phrase,
                        'train_id': train_id,
                        'error': result['message']
                    }), HTTPStatus.BAD_REQUEST.value
            except Exception as e:
                write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Save' + "," + f"macOS:{str(e)}", train_id, seq, None)
                redis_client.set_status(train_id, StatusEnum.Failed.value)
                return jsonify({
                    'status': HTTPStatus.BAD_REQUEST.phrase,
                    'train_id': train_id,
                    'error': f"macOS model merge failed: {str(e)}"
                }), HTTPStatus.BAD_REQUEST.value

    except Exception as e:
        write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Save' + "," + f"{str(e)}\n{traceback.format_exc()}", train_id, seq, None)
        redis_client.set_status(train_id, StatusEnum.Failed.value)
        return {
            'status': HTTPStatus.BAD_REQUEST.value,
            'train_id': data['trainId'],
            'error': str(e)
        }
    
from linux_code.deploy_ollama import ModelDeployerOllama
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
        
        # Execute deployment
        result = ModelDeployerOllama.deploy_model(
            train_id=data['trainId'],
            seq=data['seq'],
            status_file='status_file',
            folder_path=get_absolute_path(data['folderPath']),
            server_url=data['serverUrl'],
            model_name=data['modelName'],
            quantize=data['quantize'],
            template=data['template'],
            system=data['system'],
            parameters=data['parameters']
        )
        
        # Return response
        if result['status'] == 'success':
            write_log(LevelEnum.INFO, LogEnum.HandleEventsSuccess, 'Deploy', train_id, seq, None)
            redis_client.set_status(train_id, StatusEnum.Success.value)
            return jsonify(result), HTTPStatus.OK.value
        else:
            write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Deploy' + "," + result['message'], train_id, seq, None)
            redis_client.set_status(train_id, StatusEnum.Failed.value)
            return jsonify(result), HTTPStatus.BAD_REQUEST.value
            
    except Exception as e:
        write_log(LevelEnum.ERROR, LogEnum.HandleEventsFailed, 'Deploy' + "," + f"{str(e)}\n{traceback.format_exc()}", train_id, seq, None)
        redis_client.set_status(train_id, StatusEnum.Failed.value)
        return {
            'status': HTTPStatus.BAD_REQUEST.value,
            'train_id': data['trainId'],
            'error': str(e)
        }
    
def create_app():
    return app

if __name__ == '__main__':
    start_port=os.getenv('TRAINING_START_PORT')
    app.run(host='0.0.0.0', port=start_port
            ,threaded=True)  
