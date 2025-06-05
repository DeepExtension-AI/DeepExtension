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
import psutil
from train_callback import write_log
class Training:

    @classmethod
    def run_training(cls,train_id, params):

        write_log(LevelEnum.INFO,LogEnum.StartHandlingEvents,params.get('trainingName','Training'),train_id,params.get('seq',1),None)
        args = [
            f"--model_path={params.get('modelPath')}",
            f"--max_seq_length={params.get('maxSeqLength')}",
            f"--lora_rank={params.get('loraRank')}",
            f"--load_in_4bit={'true' if params.get('loadInFourBit') else 'false'}",
            f"--dataset_path={params.get('datasetPath')}",
            f"--max_input_length={params.get('maxInputLength')}",
            f"--max_content_length={params.get('maxContentLength')}",
            f"--max_samples={params.get('maxSamples')}",
            f"--num_generations={params.get('numGenerations')}",
            f"--max_grad_norm={params.get('maxGradNorm')}",
            f"--output_dir={params.get('outputDir')}",
            f"--max_steps={params.get('maxSteps')}",
            f"--batch_size={params.get('batchSize')}",
            f"--grad_accum_steps={params.get('gradAccumSteps')}",
            f"--learning_rate={params.get('learningRate')}",
            f"--warmup_steps={params.get('warmupSteps')}",
            f"--input_train_name={params.get('inputTrainName')}",
            f"--output_train_name={params.get('outputTrainName')}",
            f"--train_id={params.get('taskUuid')}",
            f"--seq={params.get('seq')}",
            f"--model_name={params.get('modelName')}",
        ]
        if not os.path.isfile(params.get('trainFileName')):
            write_log(LevelEnum.ERROR,LogEnum.TrainingFileNotFound,params.get('trainFileName'),train_id,params.get('seq',1),None)
            redis_client.set_status(train_id, StatusEnum.Failed.value)
            return {
                'status': 'failed',
                'train_id': train_id,
                'return_code': -1
            }
        cmd = ['python3', params.get('trainFileName')] + args
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        current_pid = psutil.Process().pid 
        host_pid = get_host_pid(current_pid)
        print(f"CurrentProcessPid: {current_pid}, HostPid: {host_pid}")
        redis_client.set_status(train_id, StatusEnum.Running.value,host_pid)
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
            write_log(LevelEnum.ERROR,LogEnum.HandleEventsFailed,params.get('trainingName','Training')+","+"Unknown",train_id,params.get('seq',1),None)
            redis_client.set_status(train_id, StatusEnum.Failed.value)
        else:
            write_log(LevelEnum.INFO,LogEnum.HandleEventsSuccess,params.get('trainingName','Training'),train_id,params.get('seq',1),None)
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


def get_host_pid(pid):
    try:
        process = psutil.Process(pid)
        return process.ppid()  
    except psutil.NoSuchProcess:
        return None

