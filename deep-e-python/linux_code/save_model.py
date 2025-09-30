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
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import torch
from train_callback import write_log
from enums import LogEnum, LevelEnum, StatusEnum

class ModelMerger:
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
    }

    @classmethod
    def merge_models(cls, train_id: str, seq: int, base_model_path: str, lora_adapter_path: str,
                   save_path: str, dtype_str: str = "bfloat16", model_usage_type: str = "chat") -> dict:
        """Core business logic for model merging
        
        Args:
            base_model_path: Path to the base model
            lora_adapter_path: Path to the LoRA adapter
            save_path: Path to save the merged model
            dtype_str: Data type string (default: "bfloat16")
            
        Returns:
            Dictionary containing operation results
        """
        try:
            dtype = cls.dtype_map.get(dtype_str.lower(), torch.bfloat16)
            # Load base model
            write_log(LevelEnum.INFO, LogEnum.MergeLoadingBaseModel, base_model_path, train_id, seq, None)
            if model_usage_type == "chat":
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path, 
                    torch_dtype=dtype
                )
            elif model_usage_type == "vision-language":
                base_model = AutoModelForVision2Seq.from_pretrained(
                    base_model_path,
                    torch_dtype=dtype,
                    device_map="auto" 
                )

            
            # Merge LoRA adapter
            write_log(LevelEnum.INFO, LogEnum.MergeAdapters, f"{base_model_path}, {lora_adapter_path}", train_id, seq, None)

            model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            model = model.merge_and_unload()
            
            # Save merged model
            write_log(LevelEnum.INFO, LogEnum.MergeSuccess, save_path, train_id, seq, None)

            model.save_pretrained(
                save_path,
                safe_serialization=True,
                max_shard_size="10GB"  # 建议分片保存大模型
            )
            
            # Save tokenizer
            if model_usage_type == "chat":
                tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
            elif model_usage_type == "vision-language":
                tokenizer = AutoProcessor.from_pretrained(lora_adapter_path)
            
            tokenizer.save_pretrained(save_path)
            return {
                "status": "success",
                "message": f"Model successfully merged and saved to {save_path}",
                "dtype_used": str(dtype),
                "model_type": model_usage_type
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

if __name__ == "__main__":
    import sys
    import json
    import argparse

    parser = argparse.ArgumentParser(description='Merge Models ')
    
    # 定义命令行参数          
    parser.add_argument('--train_id', type=str, required=True, help='训练ID')
    parser.add_argument('--seq', type=str, default='1', help='序列号')
    parser.add_argument('--save_path', type=str, required=True, help='保存路径')
    parser.add_argument('--base_path', type=str, required=True, help='基础模型路径')
    parser.add_argument('--adapter_path', type=str, required=True, help='lora路径')
    parser.add_argument('--dtype_str', type=str, default='', help='')
    parser.add_argument('--model_usage_type', type=str, default='',  help='类型')
    
    known_args, unknown_args = parser.parse_known_args()
    print("Known args:", known_args)
    print("Unknown args:", unknown_args)
    args = known_args
    
    try:
        
        # 执行部署
        result = ModelMerger.merge_models(
            train_id = args.train_id,
            seq=args.seq,
            base_model_path = args.base_path,
            lora_adapter_path = args.adapter_path,
            save_path = args.save_path,
            dtype_str=args.dtype_str,
            model_usage_type=args.model_usage_type,
        )
        
        # 输出结果
        #print(json.dumps(result))
        
        # 根据结果设置退出码
        if result.get('status') == 'success':
            print(json.dumps(result))
            sys.exit(0)
        else:
            print(json.dumps(result))
            sys.exit(1)
            
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Script execution failed: {str(e)}"
        }
        print(json.dumps(error_result))
        sys.exit(1)

