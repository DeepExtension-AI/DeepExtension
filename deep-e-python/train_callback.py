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

from transformers import TrainerCallback
import json
import time
import os
from queue import Queue
from enums import LogEnum,StatusEnum,LevelEnum
from datetime import datetime

format_str=datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")
log_file_name = f"{format_str}-deep-e-python.jsonl"

class TrainCallback(TrainerCallback):  
    def __init__(self,logging_steps,taskUuid,seq):
        super().__init__()  
        self.sse_queue = Queue()
        self.logging_steps=logging_steps
        self.metrics = []
        self.taskUuid = taskUuid
        self.seq = seq

    def on_init_end(self, args, state, control, **kwargs):
        """初始化结束时触发"""
        pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时触发"""
        pass

    def on_log(self, args, state, control, **kwargs):
        pass
    def on_train_end(self, args, state, control, **kwargs):
        pass

    def on_train_loss_report(self, train_info: dict):
        """Called to report training loss at specified intervals."""
        pass

    def on_val_loss_report(self, val_info: dict):
        """Called to report validation loss at specified intervals or the beginning."""
        pass
    def on_step_end(self, args, state, control, **kwargs):
        if not state.log_history:
            return  
        logs = state.log_history[-1]
        progress = {
            "step": state.global_step,
            "reward": logs.get('reward'),
            "reward_std": logs.get('reward_std'),
            "kl":logs.get("kl"),
            "loss":logs.get("loss"),
            "train_loss":logs.get("train_loss"),
            "validation_loss":logs.get("validation_loss"),
            "bleu":logs.get("bleu"),
            "rouge":logs.get("rouge"),
            "perplexity":logs.get("perplexity"),
            "learning_rate":logs.get("learning_rate"),
            "grad_norm":logs.get("grad_norm"),
            "epoch":logs.get("epoch")
        }
        num=(state.global_step / state.max_steps) * 100  
        write_log(LevelEnum.INFO,LogEnum.Training,f"{num}",self.taskUuid,self.seq,progress)
        
def write_log(level:LevelEnum,logEnum:LogEnum,details,train_id,seq,external_data):
    data={
        "level":level.name.lower(),
        "time":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(time.time())),
        "taskUuid":train_id,
        "seq":seq,
        "msgDetail": details,

        "msg": logEnum.value,
        "details":external_data,
    }
    print(f"[log_data]: {json.dumps(data)}")
    try:
        with open("../logs/"+log_file_name, 'a') as f:  
            f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Write Error: {e}")
