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

import subprocess
import argparse
import json
import threading

def run_local_train():
    # Define default values for all parameters
    defaults = {
        'model_path': 'your-model_path',
        'max_seq_length': '1024',
        'lora_rank': '8',
        'load_in_4bit': 'True',
        'dataset_path': 'your-dataset_path',
        'max_input_length': '1024',
        'max_content_length': '1024',
        'max_samples': '1000',
        'num_generations': '4',
        'max_grad_norm': '0.1',
        'output_dir': 'your-output_dir',
        'max_steps': '2',
        'batch_size': '2',
        'grad_accum_steps': '2',
        'learning_rate': '5e-6',
        'warmup_steps': '2',
        'input_train_name': 'prompt',
        'output_train_name': 'completion',
        'train_id': '1',
        'seq': '1',
        'status_file': 'status.json',
        'model_name': 'test'
    }

    # Build the command
    cmd = ['python3', 'your-custom-file.py'] 
    for arg, value in defaults.items():
        cmd.extend([f'--{arg}', str(value)])
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    # 实时读取并更新状态
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
if __name__ == '__main__':
    run_local_train()
