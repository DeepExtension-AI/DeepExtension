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
parser.add_argument('--load_in_4bit', type=lambda x: x.lower() == 'true', required=True)
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
        from unsloth import FastVisionModel # FastLanguageModel for LLMs
        import torch

        model, tokenizer = FastVisionModel.from_pretrained(
            MODEL_PATH,
            load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        )
        '''Insert model loading logic here'''

        print("=====================2. Dataset Preparation ========================")
        write_log(LevelEnum.INFO, LogEnum.HandleWithDataset, DATASET_PATH, args.train_id, args.seq, None)

        '''Insert dataset preparation logic here'''
        from datasets import Dataset, Features, Value, Image, Sequence
        import json


        features = Features({
            "images": Sequence({
                            "imageId": Image()
                        }),                # 自动把路径转成 PIL.Image
            InputTrainName:Sequence({
                            "question": Value("string"),
                            "answer":Value("string")
                        }),
            "generation": Value("string"),
        })
        # 加载数据
        print(DATASET_PATH)
        data = []
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        for item in data:
            if "images" in item:
                for img in item["images"]:
                    if isinstance(img["imageId"], str):
                        img["imageId"] = args.pic_relative_path +"/"+ img["imageId"] 
                        print(img["imageId"])
        # 创建数据集
        dataset = Dataset.from_list(data,features=features)
        def convert_to_conversation(sample):
            print(sample)
            conversation = [
                { "role": "user",
                "content" : [
                    {"type" : "text",  "text"  : sample[InputTrainName]["question"][0]},
                    {"type" : "image", "image" : sample["images"]['imageId'][0]} ]
                },
                { "role" : "assistant",
                "content" : [
                    {"type" : "text",  "text"  : sample[InputTrainName]["answer"][0]} ]
                },
            ]
            return { "messages" : conversation }
        pass

        converted_dataset = [convert_to_conversation(sample) for sample in dataset]
        converted_dataset[0]
        print("=====================3. Trainer Configuration ========================")
        write_log(LevelEnum.INFO, LogEnum.ConfigTrainingParams, None, args.train_id, args.seq, None)

        '''Insert trainer configuration logic here'''
        from unsloth.trainer import UnslothVisionDataCollator
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers     = True, # False if not finetuning vision layers
            finetune_language_layers   = True, # False if not finetuning language layers
            finetune_attention_modules = True, # False if not finetuning attention layers
            finetune_mlp_modules       = True, # False if not finetuning MLP layers

            r = 16,           # The larger, the higher the accuracy, but might overfit
            lora_alpha = 16,  # Recommended alpha == r at least
            lora_dropout = 0,
            bias = "none",
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )
        from trl import SFTTrainer,SFTConfig
        from unsloth.trainer import UnslothVisionDataCollator

        FastVisionModel.for_training(model) # Enable for training!

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
            train_dataset = converted_dataset,
            args = SFTConfig(
                per_device_train_batch_size = BATCH_SIZE,
                gradient_accumulation_steps = GRAD_ACCUM_STEPS,
                warmup_steps = WARMUP_STEPS,
                **({"max_steps": MAX_STEPS} if MAX_STEPS is not None else {}),
                # num_train_epochs = 1, # Set this instead of max_steps for full training runs
                learning_rate = LEARNING_RATE,
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none",     # For Weights and Biases

                # You MUST put the below items for vision finetuning:
                remove_unused_columns = False,
                dataset_text_field = "",
                dataset_kwargs = {"skip_prepare_dataset": True},
                dataset_num_proc = 4,
                max_seq_length = 2048,
            ),
            callbacks=[callback],
        )
        print("=====================4. Training Started ========================")
        write_log(LevelEnum.INFO, LogEnum.StartTraining, None, args.train_id, args.seq, None)
        '''Insert training logic here'''
        trainer_stats = trainer.train()
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