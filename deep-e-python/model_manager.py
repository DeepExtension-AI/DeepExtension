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

import logging
import time
import threading
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextStreamer
import torch
import traceback

class ModelManager:
    def __init__(self):
        self.models: Dict[str, dict] = {}  
        self.last_used: Dict[str, datetime] = {}  
        self.lock = threading.Lock() 
        self.conversation_histories = {}  
        self._start_cleanup_thread()  

    def _start_cleanup_thread(self):
        def cleanup():
            while True:
                self._check_and_unload_idle_models()
                time.sleep(60) 

        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()

    def _check_and_unload_idle_models(self):
        with self.lock:
            now = datetime.now()
            models_to_unload = []
            
            for model_name, last_used in self.last_used.items():
                if (now - last_used) > timedelta(seconds=10):
                    models_to_unload.append(model_name)
            
            for model_name in models_to_unload:
                self._unload_model_internal(model_name)

    def _unload_model_internal(self, model_name: str):
        if model_name in self.models:
            if "model" in self.models[model_name]:
                del self.models[model_name]["model"]
            if "tokenizer" in self.models[model_name]:
                del self.models[model_name]["tokenizer"]
            del self.models[model_name]
            if model_name in self.last_used:
                del self.last_used[model_name]
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _update_last_used(self, model_name: str):
        with self.lock:
            self.last_used[model_name] = datetime.now()

    def load_model(
        self,
        model_id: str,
        model_name: str,
        baseModelPath: str,
        device: Optional[str] = "cuda",
        precision: Optional[str] = "fp16",
        merge_lora: bool = True 
    ) -> bool:
        try:
            import os
            os.environ["UNSLOTH_DISABLE_AUTO_OPTIMIZE"] = "1"
            if model_name in self.models:
                self._update_last_used(model_name)
                return True
            model_args = {
                "pretrained_model_name_or_path": baseModelPath,
                "trust_remote_code": True,
                "device_map": "auto" if device == "cuda" else None
            }
            if precision == "fp16":
                model_args["torch_dtype"] = torch.float16
            elif precision == "fp32":
                model_args["torch_dtype"] = torch.float32
            config = AutoConfig.from_pretrained(baseModelPath)
            tokenizer = AutoTokenizer.from_pretrained(baseModelPath)
            model = AutoModelForCausalLM.from_pretrained(**model_args)
            if merge_lora and hasattr(model, "peft_config"):
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, model_id)
                model = model.merge_and_unload() 
            if not hasattr(config, 'max_seq_length'):
                config.max_seq_length = getattr(config, 'max_position_embeddings', 2048)
            if not hasattr(model, 'max_seq_length'):
                model.max_seq_length = config.max_seq_length
            with self.lock:
                self.models[model_name] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "config": config,
                    "device": device
                }
                self.last_used[model_name] = datetime.now()
            return True
        except Exception as e:
            traceback.print_exc()  
            return False

    def get_model_info(self, model_name: str) -> Optional[dict]:
        if model_name not in self.models:
            return None
        
        self._update_last_used(model_name)
        
        config = self.models[model_name]["config"].to_dict()
        return {
            "model_type": config.get("model_type"),
            "num_parameters": sum(p.numel() for p in self.models[model_name]["model"].parameters()),
            "max_length": config.get("max_position_embeddings", 2048),
            "vocab_size": config.get("vocab_size"),
            "device": self.models[model_name]["device"]
        }
    
    def generate_text(
        self,
        model_name: str,
        prompt: str,
        max_length: int ,
        max_context_tokens: int ,
        temperature: float ,
        top_p: float,
        top_k: int,
        system_prompt: Optional[str] ,
        reset_context: bool ,
        session_id: str,
        history: Optional[List[Dict]]
    ) -> Tuple[str, List[Dict]]:
        """
        Generate text and maintain conversation context

        Parameters:
            model_name: Name of the model to use
            prompt: User input prompt
            session_id: Conversation ID for maintaining distinct contexts
            max_length: Maximum length of generated text (in tokens)
            max_context_tokens: Maximum context window size (in tokens)
            temperature: Generation temperature (0-1, higher = more random)
            top_p: Nucleus sampling probability threshold (0-1)
            top_k: Top-k sampling (keep only top k probability tokens)
            system_prompt: System prompt/instruction
            reset_context: Whether to reset the conversation context

        Returns:
            Tuple[generated_text, updated_context]
        """
        """
        Generate text and update last used timestamp
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        self._update_last_used(model_name)
        model = self.models[model_name]["model"]
        tokenizer = self.models[model_name]["tokenizer"]
        if history is not None:
            context = history.copy()
            if system_prompt:
                print(system_prompt)
        else:
            if reset_context or session_id not in self.conversation_histories:
                self.conversation_histories[session_id] = []
                if system_prompt:
                    self.conversation_histories[session_id].append({
                        "role": "system", 
                        "content": system_prompt
                    })
            context = self.conversation_histories[session_id].copy()
        messages=self._format_context(context)
        try:
            inputs = tokenizer(messages, return_tensors="pt", truncation=True, max_length=max_context_tokens).to(model.device)
            if hasattr(model,"past_key_values"):
                inputs["past_key_values"]=None
            streamer = TextStreamer(tokenizer)
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True if temperature>0 else False,
                use_cache=True,
                streamer=streamer, 
            )
            input_length = inputs.input_ids.shape[1]
            generated_ids = outputs[0][input_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if "assistant:" in response:
                assistant_response = response.split("assistant:")[-1].strip()
            else:
                assistant_response = response.strip()
            context.append({"role": "assistant", "content": assistant_response})
            tmp={
                "text": assistant_response,
                "status": "resp"
            }
            print(json.dumps(tmp,ensure_ascii=False))
            return 
        except Exception as e:
            traceback.print_exc()  
            return {
                "error": str(e),
                "status": "error",
                "context": context if 'context' in locals() else []
            }

    def _format_context(self, context: List[Dict]) -> str:
        
        prompt_parts = []

        for message in context:
            role = message["role"]
            content = message["content"]
    
            if role in ["system", "user", "assistant"]:
                    prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if not prompt_parts[-1].startswith("<|im_start|>assistant"):
            prompt_parts.append("<|im_start|>assistant")
        prompt = "\n".join(prompt_parts)
        return prompt
        
    
    def clear_context(self, session_id: str = "default"):
        if session_id in self.conversation_histories:
            self.conversation_histories[session_id] = []
    def unload_model(self, model_name: str):
        with self.lock:
            self._unload_model_internal(model_name)
            
import argparse
import json
import sys
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--base_model_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--session_id', type=str, default='default')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_context_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--system_prompt', type=str)
    parser.add_argument('--reset_context', type=str)
    parser.add_argument('--history', type=str)

    args = parser.parse_args()

    try:
        model_manager = ModelManager()
        
        success = model_manager.load_model(
            model_id=args.model_id,
            model_name=args.model_name,
            baseModelPath=args.base_model_path
        )
        
        if not success:
            raise Exception("Loading Model Failed")

        history = json.loads(args.history) if args.history else None

        result = model_manager.generate_text(
            model_name=args.model_name,
            prompt=args.prompt,
            session_id=args.session_id,
            max_length=args.max_length,
            max_context_tokens=args.max_context_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            system_prompt=args.system_prompt,
            reset_context=args.reset_context == 'true' if args.reset_context else False,
            history=history
        )

        model_manager._start_cleanup_thread()
        
        # 输出结果
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }, ensure_ascii=False))
        sys.exit(1)

if __name__ == '__main__':
    main()
