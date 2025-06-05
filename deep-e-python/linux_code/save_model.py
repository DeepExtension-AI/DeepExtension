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

from transformers import AutoModelForCausalLM, AutoTokenizer
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
                   save_path: str, dtype_str: str = "bfloat16") -> dict:
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
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path, 
                torch_dtype=dtype
            )
            
            # Merge LoRA adapter
            write_log(LevelEnum.INFO, LogEnum.MergeAdapters, base_model_path + "," + lora_adapter_path, train_id, seq, None)
            model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            model = model.merge_and_unload()
            
            # Save merged model
            write_log(LevelEnum.INFO, LogEnum.MergeSuccess, save_path, train_id, seq, None)
            model.save_pretrained(save_path, safe_serialization=True)
            
            # Save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
            tokenizer.save_pretrained(save_path)
            return {
                "status": "success",
                "message": f"Model successfully merged and saved to {save_path}",
                "dtype_used": str(dtype)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }