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
from train_callback import TrainCallback,StatusEnum
from redis_util import redis_client
import traceback
import argparse
import os
import torch
from train_callback import TrainCallback,write_log,StatusEnum,LevelEnum,LogEnum
global MODEL_PATH, MAX_SEQ_LENGTH, LORA_RANK, LOAD_IN_4BIT
global DATASET_PATH, MAX_INPUT_LENGTH, MAX_CONTENT_LENGTH, MAX_SAMPLES
global NUM_GENERATIONS, MAX_GRAD_NORM, OUTPUT_DIR, MAX_STEPS
global BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE, WARMUP_STEPS,InputTrainName,OutputTrainName
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--max_seq_length', type=int, required=True)
parser.add_argument('--lora_rank', type=int, required=True)
parser.add_argument('--load_in_4bit', type=lambda x: x.lower() == 'true',  required=True)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--max_input_length', type=int, required=True)
parser.add_argument('--max_content_length', type=int, required=True)
parser.add_argument('--max_samples', type=int, required=True)
parser.add_argument('--num_generations', type=int, required=True)
parser.add_argument('--max_grad_norm', type=float, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--max_steps', type=int)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--grad_accum_steps', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--warmup_steps', type=int, required=True)
parser.add_argument('--input_train_name', type=str)
parser.add_argument('--output_train_name', type=str)
parser.add_argument('--train_id', type=str, required=True)
parser.add_argument('--seq', type=int, required=True)
parser.add_argument('--model_name',type=str,required=True)
known_args, unknown_args = parser.parse_known_args()
print("Known args:", known_args)
print("Unknown args:", unknown_args)
args = known_args
callback = TrainCallback(1,args.train_id,args.seq)
# 设置全局变量
MODEL_PATH = args.model_path
MAX_SEQ_LENGTH = args.max_seq_length
LORA_RANK = args.lora_rank
LOAD_IN_4BIT = args.load_in_4bit
DATASET_PATH = args.dataset_path
MAX_INPUT_LENGTH = args.max_input_length
MAX_CONTENT_LENGTH = args.max_content_length
MAX_SAMPLES = args.max_samples
NUM_GENERATIONS = args.num_generations
MAX_GRAD_NORM = args.max_grad_norm
OUTPUT_DIR = args.output_dir
MAX_STEPS = args.max_steps
BATCH_SIZE = args.batch_size
GRAD_ACCUM_STEPS = args.grad_accum_steps
LEARNING_RATE = args.learning_rate
WARMUP_STEPS = args.warmup_steps
InputTrainName = args.input_train_name
OutputTrainName = args.output_train_name
MODEL_NAME=args.model_name
#============ from here add your own train code
import subprocess
import argparse
import threading

defaults = {
    "model": os.path.abspath(MODEL_PATH),
    "fine-tune-type": "lora",
    "optimizer": "adamw",
    "data": DATASET_PATH,
    "seed": 0,
    "num-layers": 16,
    "batch-size": BATCH_SIZE,
    "iters": MAX_STEPS,
    "val-batches": 25,
    "learning-rate": LEARNING_RATE,
    "steps-per-report": 10,  ##LOG SETPS
    "steps-per-eval": 100,
    "adapter-path": OUTPUT_DIR,
    "save-every": 100,
    "test-batches": 500,
    "max-seq-length": MAX_SEQ_LENGTH
}

cmd = ['python3', '-m', 'mlx_lm','lora']
for arg, value in defaults.items():
    cmd.extend([f'--{arg}', str(value)])
cmd.extend(['--train'])
process = subprocess.Popen(
    cmd, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE,
    text=True
)
# Read and update status in real-time
def print_real(pipe, is_error=False):
    """Print output lines with error prefix if needed"""
    for line in iter(pipe.readline, ''):
        print(f"{'[ERR] ' if is_error else '[OUT] '}{line.rstrip()}")

# Create threads to handle output
stdout_thread = threading.Thread(
    target=print_real,
    args=(process.stdout, False)
)
stderr_thread = threading.Thread(
    target=print_real,
    args=(process.stderr, True)
)

# Start output monitoring threads
stdout_thread.start()
stderr_thread.start()

# Wait for process completion
return_code = process.wait()

# Wait for output threads to finish
stdout_thread.join()
stderr_thread.join()

# Read remaining output
stdout_lines = process.stdout.readlines()
stderr_lines = process.stderr.readlines()

# Combine all output
all_output = stdout_lines + stderr_lines

# Check for tracebacks in output
if any("Traceback" in line for line in all_output):
    # Get last non-empty line as error message
    for line in reversed(all_output):
        if line.strip():  # Skip empty lines
            print_real(line.strip(), True)
            break

# Close pipes
process.stdout.close()
process.stderr.close()
print(f"return_code:{return_code}")
if return_code != 0:
    write_log(LevelEnum.ERROR,LogEnum.TrainingFailed,None,args.train_id,args.seq,None)
    redis_client.set_status(args.train_id, StatusEnum.Failed.value)
#============ end here with your own train code