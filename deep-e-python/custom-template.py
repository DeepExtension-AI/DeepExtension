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

import argparse
from train_callback import TrainCallback,write_log,StatusEnum,LevelEnum,LogEnum
import time
import traceback
import os
import torch

global MODEL_PATH, MAX_SEQ_LENGTH, LORA_RANK, LOAD_IN_4BIT
global DATASET_PATH, MAX_INPUT_LENGTH, MAX_CONTENT_LENGTH, MAX_SAMPLES
global NUM_GENERATIONS, MAX_GRAD_NORM, OUTPUT_DIR, MAX_STEPS
global BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE, WARMUP_STEPS,InputTrainName,OutputTrainName
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--max_seq_length', type=int, required=True)
parser.add_argument('--lora_rank', type=int, required=True)
parser.add_argument('--load_in_4bit', type=bool, required=True)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--max_input_length', type=int, required=True)
parser.add_argument('--max_content_length', type=int, required=True)
parser.add_argument('--max_samples', type=int, required=True)
parser.add_argument('--num_generations', type=int, required=True)
parser.add_argument('--max_grad_norm', type=float, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--max_steps', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--grad_accum_steps', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--warmup_steps', type=int, required=True)
parser.add_argument('--input_train_name', type=str, required=True)
parser.add_argument('--output_train_name', type=str, required=True)
parser.add_argument('--train_id', type=str, required=True)
parser.add_argument('--seq', type=int, required=True)
parser.add_argument('--model_name',type=str,required=True)
args = parser.parse_args()

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
callback = TrainCallback(1,args.train_id,args.seq)

#============ from here add your own train code


#============ end here with your own train code

def training(): 
    global trainer
    global model
    global tokenizer
    global InputTrainName
    global OutputTrainName

    try:

        print("=====================1. Model Initialization ========================")
        # During model initialization
        write_log(LevelEnum.INFO, LogEnum.InitializingModel, MODEL_PATH, args.train_id, args.seq, None)

        '''Insert model loading logic here'''

        print("=====================2. Dataset Preparation ========================")
        write_log(LevelEnum.INFO, LogEnum.HandleWithDataset, DATASET_PATH, args.train_id, args.seq, None)

        '''Insert dataset preparation logic here'''

        print("=====================3. Trainer Configuration ========================")
        write_log(LevelEnum.INFO, LogEnum.ConfigTrainingParams, None, args.train_id, args.seq, None)

        '''Insert trainer configuration logic here'''

        print("=====================4. Training Started ========================")
        write_log(LevelEnum.INFO, LogEnum.StartTraining, None, args.train_id, args.seq, None)

        '''Insert training logic here'''
        trainer.train()

        print("=====================5. Saving Model ========================")
        '''Insert model saving logic here'''
        write_log(LevelEnum.INFO, LogEnum.TrainingSuccess, None, args.train_id, args.seq, None)
        write_log(LevelEnum.INFO, LogEnum.SaveTrainedModel, None, args.train_id, args.seq, None)

        model.save_pretrained(OUTPUT_DIR)     # Save LoRA weights
        tokenizer.save_pretrained(OUTPUT_DIR) # Save tokenizer configuration

        print("=====================6. Training Completed ========================")
        write_log(LevelEnum.INFO, LogEnum.SaveTrainedModelSuccess, OUTPUT_DIR, args.train_id, args.seq, None)
       
    except Exception as e:
        error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        write_log(LevelEnum.ERROR,LogEnum.TrainingFailed,None,str(e),args.seq,0)
        raise
    finally:
        del InputTrainName
        del OutputTrainName
        if 'trainer' in locals():
            del trainer
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if 'dataset' in locals():
            del dataset
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    training()
