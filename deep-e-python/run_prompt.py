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

from enums import StatusEnum
from redis_util import redis_client
import threading
import subprocess
class Prompting:
    @classmethod
    def run_prompting(cls, params):
        import json
        import traceback
        try:
            instruction = [
                "conda", "run", "-n", params.get('condaEnv','base'),
                "python", params.get('promptFile','qwen_inference')+".py",
            ]
            modelType= params.get('modelType', 'chat') or 'chat'
            chat_args = [
                f"--model_id={params.get('modelId', '0')}",
                f"--model_name={params['modelName']}",
                f"--base_model_path={params['baseModelPath']}",
                f"--prompt={params['prompt']}",
                f"--session_id={params.get('sessionId', 'default')}",
                f"--max_length={params.get('maxLength', 512)}",
                f"--max_context_tokens={params.get('maxContentTokens', 2048)}",
                f"--temperature={params.get('temperature', 0.7)}",
                f"--top_p={params.get('topP', 0.9)}",
                f"--top_k={params.get('topK', 50)}",
                f"--model_type={modelType}",
                f"--template={params.get('template','')}"
            ]

            # Add optional parameters
            if 'systemPrompt' in params:
                chat_args.append(f"--system_prompt={params['systemPrompt']}")
            if 'resetContext' in params:
                chat_args.append(f"--reset_context={str(params['resetContext']).lower()}")
            if 'history' in params:
                chat_args.append(f"--history={json.dumps(params['history'])}")
            cmd = instruction + chat_args
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
                        print(f"stderr1:{tmp}")
                        ##yield f"data: {tmp}\n\n"
                except:
                    tmp = json.dumps({"finish_reason": stderr})
                    print(f"stderr2:{tmp}")
                    ##yield f"data: {tmp}\n\n"
            else:
                
                tmp = json.dumps({"finish_reason": stderr, "text": ""})
                print(f"stderr3:{tmp}")
                ##yield f"data: {tmp}\n\n"

        except Exception as e:
            error_msg = f"Chat process execution failed: {str(e)}\n{traceback.format_exc()}"
            tmp = json.dumps({"finish_reason": error_msg, "text": ""}) 
            yield f"data: {tmp}\n\n"
            pass

    @classmethod
    def run_generate_images(cls,train_id, params):
        conda_env = params.get('condaEnv', 'sd') or 'sd'
        cmd = [
            "conda", "run", "-n", conda_env,
            "python", params.get('promptFile')+".py",
            "--model_path", params.get('model') ,
            "--lora_path", params.get('loraPath') ,
            "--prompt", params.get('prompt') ,
            "--negative_prompt",  params.get('negativePrompt') ,
            "--output" ,f"../images/{train_id}.png" ,
            "--batch_size", params.get('batchSize') ,
            "--aspect_ratio", params.get('aspectRatio') ,
        ]
        print(f"Executing command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        redis_client.set_status(train_id, StatusEnum.Running.value,0)
        def printReal(pipe, is_error=False):
            for line in iter(pipe.readline, ''):
                print(f"{'[ERR] ' if is_error else '[OUT] '}{line.rstrip()}")

        stdout_thread = threading.Thread(target=printReal, args=(process.stdout, False))
        stderr_thread = threading.Thread(target=printReal, args=(process.stderr, True))

        stdout_thread.start()
        stderr_thread.start()

        return_code = process.wait()

        stdout_thread.join()
        stderr_thread.join()
        stdout_lines = process.stdout.readlines()
        stderr_lines = process.stderr.readlines()
        
        all_output = stdout_lines + stderr_lines
        if any("Traceback" in line for line in all_output):
            for line in reversed(all_output):
                if line.strip(): 
                    printReal(line.strip(),True)

        process.stdout.close()
        process.stderr.close()
        if return_code == 0:
            redis_client.set_status(train_id, StatusEnum.Success.value)
        else:
            redis_client.set_status(train_id, StatusEnum.Failed.value)
