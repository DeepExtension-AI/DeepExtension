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

from enums import StatusEnum,LogEnum,LevelEnum
from redis_util import redis_client
import threading
import subprocess
import os
import json
from train_callback import write_log
class Training:

    @classmethod
    def run_training(cls,train_id, params):

        write_log(LevelEnum.INFO,LogEnum.StartHandlingEvents,params.get('training_name','Training'),train_id,params.get('seq',1),None)
        args = []
        for key, value in params.items():
            if value is not None:  
        ##modelName   model_name
                if isinstance(value, bool):
                    arg_value = 'true' if value else 'false'
                elif isinstance(value, (list, dict)):
                    arg_value = json.dumps(value)  
                else:
                    arg_value = str(value)
                
                args.append(f"--{key}={arg_value}")
        if not os.path.isfile(params.get('train_file_name')):
            write_log(LevelEnum.ERROR,LogEnum.TrainingFileNotFound,params.get('train_file_name'),train_id,params.get('seq',1),None)
            redis_client.set_status(train_id, StatusEnum.Failed.value)
            return {
                'status': 'failed',
                'train_id': train_id,
                'return_code': -1
            }
        print(args)
        cmd = ['python3', params.get('train_file_name')] + args
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        current_pid = process.pid 
        host_pid = get_host_pid(current_pid)
        print(f"CurrentProcessPid: {current_pid}, HostPid: {host_pid}")
        redis_client.set_status(train_id, StatusEnum.Running.value,current_pid)
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
        print(f"return_code{return_code}")
        if return_code != 0:
            write_log(LevelEnum.ERROR,LogEnum.HandleEventsFailed,params.get('train_name','Training')+","+"Unknown",train_id,params.get('seq',1),None)
            redis_client.set_status(train_id, StatusEnum.Failed.value)
        else:
            write_log(LevelEnum.INFO,LogEnum.HandleEventsSuccess,params.get('train_name','Training'),train_id,params.get('seq',1),None)
            data=redis_client.get_status(train_id)
            if (data.get("status") != StatusEnum.Success.value and 
                data.get("status") != StatusEnum.Failed.value):
                redis_client.set_status(train_id, StatusEnum.Success.value)
            ##TODO
        return {
            'status': 'completed' if return_code == 0 else 'failed',
            'train_id': train_id,
            'return_code': return_code
        }


def get_host_pid(container_pid):
    try:
        with open(f'/proc/{container_pid}/status') as f:
            for line in f:
                if line.startswith('NSpid'):
                    parts = line.split()
                    if len(parts) >= 3:
                        return int(parts[2])
    except Exception as e:
        print(f"Error getting host PID: {e}")
        return None
