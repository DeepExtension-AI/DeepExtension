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

import os
import sys

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)




from train_callback import StatusEnum,write_log,StatusEnum,LevelEnum,LogEnum
from redis_util import redis_client
import argparse

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
parser.add_argument('--pic_relative_path',type=str)
known_args, unknown_args = parser.parse_known_args()
print("Known args:", known_args)
print("Unknown args:", unknown_args)
args = known_args
#callback = TrainCallback(1,args.train_id,args.seq)
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
OUTPUT_DIR = f"../{args.output_dir}"
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
    "--resume_from_checkpoint": "latest" ,#从最新的检查点恢复训练
    "--seed": 42, #随机种子，用于确保实验可重复性
    "--disable_benchmark": True, # 不禁用基准测试(启用性能测量)
    "--pretrained_model_name_or_path": os.path.abspath(MODEL_PATH), #预训练模型路径
    "--output_dir": OUTPUT_DIR,  #训练输出和模型保存路径
    "--model_family":"sd3", #"sdxl", "flux", "sd3", "sana" 模型家族(Stable Diffusion 3)
    "--model_type": "lora", #lora--full
    "--lora_type": "standard" ,#标准LoRA类型
    "--attention_mechanism": "diffusers",# 使用diffusers库的注意力机制
    "--max_train_steps": MAX_STEPS ,#最大训练步数(epochs为0时以此为准)
    "--num_train_epochs": 0 ,  #训练epoch数(0表示使用步数)
    "--train_batch_size": BATCH_SIZE, #训练批量大小 1
    "--gradient_checkpointing": True ,  #启用梯度检查点(节省显存)
    "--mixed_precision": "bf16" ,   #使用bfloat16混合精度训练
    "--optimizer": "adamw_bf16",  #使用AdamW优化器(bf16版本)
    "--learning_rate": LEARNING_RATE,  #学习率
    "--lr_scheduler": "polynomial",#多项式学习率调度器
    "--lr_warmup_steps": WARMUP_STEPS ,  #学习率预热步数
    "--data_backend_config": f"./config/multidatabackend_{args.train_id}.json", #数据后端配置
    "--aspect_bucket_rounding": 2 ,  #图像宽高比分桶的舍入参数
    "--minimum_image_size": 0 ,# 最小图像尺寸(0表示无限制)
    "--resolution": 1024 ,# 训练分辨率
    "--resolution_type": "pixel_area" ,# 分辨率类型(按像素面积)
    "--caption_dropout_probability": 0.05 ,# 标题丢弃概率(数据增强)
    #"--validation_disable": True,
    "--validation_steps": 500 ,# 每500步验证一次
    "--validation_seed": 42,#验证使用的随机种子
    "--validation_resolution": "1024x1024" ,#验证图像分辨率
    "--validation_guidance": 5.0, #验证时的CFG guidance scale
    "--validation_prompt": "A photo-realistic image of a cat" ,  #验证提示词
    "--validation_num_inference_steps": "20" ,#验证时的推理步数
    "--validation_torch_compile": False ,#不启用torch编译加速验证
    "--checkpointing_steps": 500 ,#每500步保存检查点
    "--checkpoints_total_limit": 5 ,#最多保留5个检查点
 #覆盖数据集新配置
}

# dataloader configuration
resolution_configs = {
    256: {"resolution": 256, "minimum_image_size": 128},
    512: {"resolution": 512, "minimum_image_size": 256},
    768: {"resolution": 768, "minimum_image_size": 512},
    1024: {"resolution": 1024, "minimum_image_size": 768},
    1440: {"resolution": 1440, "minimum_image_size": 1024},
    2048: {"resolution": 2048, "minimum_image_size": 1440},
}
default_dataset_configuration = {
    "id": "PLACEHOLDER",
    "type": "local",
    "instance_data_dir": None,
    "crop": False,
    "resolution_type": "pixel_area",
    "metadata_backend": "discovery",
    "caption_strategy": "filename",
    "cache_dir_vae": "vae",
}
default_cropped_dataset_configuration = {
    "id": "PLACEHOLDER-crop",
    "type": "local",
    "instance_data_dir": None,
    "crop": True,
    "crop_aspect": "square",
    "crop_style": "center",
    "vae_cache_clear_each_epoch": False,
    "resolution_type": "pixel_area",
    "metadata_backend": "discovery",
    "caption_strategy": "filename",
    "cache_dir_vae": "vae-crop",
}

default_local_configuration = [
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": True,
        "type": "local",
        "cache_dir": "text",
        "write_batch_size": 128,
    },
]
def create_dataset_config(resolution, default_config):
    dataset = default_config.copy()
    dataset.update(resolution_configs[resolution])
    dataset["id"] = f"{dataset['id']}-{resolution}"
    dataset["instance_data_dir"] = os.path.abspath(args.pic_relative_path)
    dataset["repeats"] = 10
    # we want the absolute path, as this works best with datasets containing nested subdirectories.
    dataset["cache_dir_vae"] = os.path.abspath(
        os.path.join(
            "cache/",
            "sd3",
            dataset["cache_dir_vae"],
            str(resolution),
        )
    )
    dataset["id"] = dataset["id"].replace("PLACEHOLDER", args.train_id)
    dataset["caption_strategy"] = "textfile"


    return dataset

# this is because the text embed dataset is in the default config list at the top.
# it's confusingly written because i'm lazy, but you could do this any number of ways.
default_base_resolutions = "1024"
dataset_resolutions = [int(default_base_resolutions)]
default_local_configuration[0]["cache_dir"] = os.path.abspath(
    os.path.join("cache/", "sd3", "text")
)
for resolution in dataset_resolutions:
    uncropped_dataset = create_dataset_config(
        resolution, default_dataset_configuration
    )
    default_local_configuration.append(uncropped_dataset)
    cropped_dataset = create_dataset_config(
        resolution, default_cropped_dataset_configuration
    )
    default_local_configuration.append(cropped_dataset)

print("Dataloader configuration:")
print(default_local_configuration)


import json
with open('SimpleTuner/config/config.json', 'w') as f:
    json.dump(defaults, f, indent=4)
with open(f"SimpleTuner/config/multidatabackend_{args.train_id}.json", 'w') as f:
    json.dump(default_local_configuration, f, indent=4)
        
import subprocess
import logging
import threading

def stream_reader(stream, logger):
    """实时读取输出流，避免阻塞"""
    for line in iter(stream.readline, ''):
        logger.info(line.strip())

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Subprocess')
cmd = ['bash', 'train.sh']
cwd_path = './SimpleTuner'

process = subprocess.Popen(
    cmd,
    cwd=cwd_path,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,  # 行缓冲
    close_fds=True  # 关键修复 [:ml-citation{ref="3" data="citationList"}6]^
)

# 启动独立线程处理输出
threading.Thread(target=stream_reader, args=(process.stdout, logger)).start()
threading.Thread(target=stream_reader, args=(process.stderr, logger)).start()

# 等待进程结束
return_code = process.wait()
logger.info(f"进程结束，返回码: {return_code}")
print(f"return_code:{return_code}")
if return_code != 0:
    write_log(LevelEnum.ERROR,LogEnum.TrainingFailed,None,args.train_id,args.seq,None)
    redis_client.set_status(args.train_id, StatusEnum.Failed.value)
#============ end here with your own train code