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

    # 模型参数
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


# -*- coding: utf-8 -*-
"""Qwen2.5_(3B)-GRPO.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb

To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
<div class="align-center">
<a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
<a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
<a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
</div>

To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://docs.unsloth.ai/get-started/installing-+-updating).

You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)

### News

Read our **[Qwen3 Guide](https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune)** and check out our new **[Dynamic 2.0](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs)** quants which outperforms other quantization methods!

Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks).

### Installation
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth vllm
# else:
#     # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
#     !pip install --no-deps unsloth vllm

# Commented out IPython magic to ensure Python compatibility.
# #@title Colab Extra Install { display-mode: "form" }
# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth vllm
# else:
#     !pip install --no-deps unsloth vllm
#     # [NOTE] Do the below ONLY in Colab! Use [[pip install unsloth vllm]]
#     # Skip restarting message in Colab
#     import sys, re, requests; modules = list(sys.modules.keys())
#     for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None
#     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft "trl==0.15.2" triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
# 
#     # vLLM requirements - vLLM breaks Colab due to reinstalling numpy
#     f = requests.get("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/requirements/common.txt").content
#     with open("vllm_requirements.txt", "wb") as file:
#         file.write(re.sub(rb"(transformers|numpy|xformers)[^\n]{1,}\n", b"", f))
#     !pip install -r vllm_requirements.txt

"""### Unsloth

Load up `Qwen 2.5 3B Instruct`, and set parameters
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
#max_seq_length = 1024 # Can increase for longer reasoning traces
#lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = LOAD_IN_4BIT, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = LORA_RANK,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = LORA_RANK,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

"""### Data Prep
<a name="Data"></a>

We directly leverage [@willccbb](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) for data prep and all reward functions. You are free to create your own!
"""

import re
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    #data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    #DATASET_FILE_PATH = "openai-gsm8k_first_100_lines.jsonl"
    data = load_dataset(    
            "json",  # 指定文件类型
            data_files=DATASET_PATH,
            split="train" )
    if MAX_SAMPLES > 0:
        data = data.select(range(min(MAX_SAMPLES, len(data))))
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

"""<a name="Train"></a>
### Train the model

Now set up GRPO Trainer and all configurations!
"""

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = LEARNING_RATE,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM_STEPS, # Increase to 4 for smoother training
    num_generations = NUM_GENERATIONS, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = MAX_STEPS,
    save_steps = 250,
    max_grad_norm = MAX_GRAD_NORM,
    report_to = "none", # Can use Weights & Biases
#    output_dir = "outputs",
)

"""And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!

You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!

| Step | Training Loss | reward    | reward_std | completion_length | kl       |
|------|---------------|-----------|------------|-------------------|----------|
| 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
| 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
| 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |

"""

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
    callbacks=[callback],
)

"""Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp or a UI based system like Jan or Open WebUI. You can install Jan [here](https://github.com/janhq/jan) and Open WebUI [here](https://github.com/open-webui/open-webui)

And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

Some other links:
1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!

<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
</div>

"""



#============ end here with your own train code

def training(): 
	#====================== 接收参数 ======================
    global trainer
    global model
    global tokenizer
    global InputTrainName
    global OutputTrainName

    try:
        print("=====================1. Initialize Model ========================")
        # When initializing the model
        write_log(LevelEnum.INFO, LogEnum.InitializingModel, MODEL_PATH, args.train_id, args.seq, None)

        '''Add model initialization logic here'''

        print("=====================2. Prepare Dataset ========================")
        write_log(LevelEnum.INFO, LogEnum.HandleWithDataset, DATASET_PATH, args.train_id, args.seq, None)

        '''Add dataset preparation logic here'''

        print("=====================3. Configure Trainer ========================")
        write_log(LevelEnum.INFO, LogEnum.ConfigTrainingParams, None, args.train_id, args.seq, None)

        '''Add trainer configuration logic here'''

        print("=====================4. Start Training ========================")
        write_log(LevelEnum.INFO, LogEnum.StartTraining, None, args.train_id, args.seq, None)

        '''Add training logic here'''
        trainer.train()

        print("=====================5. Saving Model ========================")
        '''Add saving logic here'''
        write_log(LevelEnum.INFO, LogEnum.TrainingSuccess, None, args.train_id, args.seq, None)
        write_log(LevelEnum.INFO, LogEnum.SaveTrainedModel, None, args.train_id, args.seq, None)

        model.save_pretrained(OUTPUT_DIR)     # Save LoRA weights
        tokenizer.save_pretrained(OUTPUT_DIR) # Save tokenizer config

        print("=====================6. Training Completed ========================")
        write_log(LevelEnum.INFO, LogEnum.SaveTrainedModelSuccess, OUTPUT_DIR, args.train_id, args.seq, None)
       
    except Exception as e:
        error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        write_log(LevelEnum.ERROR,LogEnum.TrainingFailed,str(e),args.train_id,args.seq,None)
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
