from pathlib import Path
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import fetch_from_hub, get_model_path,save
from mlx_lm.tuner.utils import dequantize, load_adapters
import os
from train_callback import write_log
from enums import LogEnum,LevelEnum,StatusEnum
current_dir = os.path.dirname(os.path.abspath(__file__))

def merge_model(
        train_id :str ,seq :int,
    base_model_path: str,
    adapter_path: str,
    save_path: str,
    dequantize_model: bool = False,
):
    try:
        relative_base_path = os.path.relpath(base_model_path, current_dir)
        write_log(LevelEnum.INFO,LogEnum.MergeLoadingBaseModel,relative_base_path,train_id,seq,None)
        model_path = Path(base_model_path)
         
        model, config, tokenizer = fetch_from_hub(model_path)
    
        model.freeze()
        model = load_adapters(model, adapter_path)
    
        fused_linears = [(n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")]
        if fused_linears:
            model.update_modules(tree_unflatten(fused_linears))
    
        # Whether to dequantize
        if dequantize_model:
            write_log(LevelEnum.INFO, LogEnum.MergeModelQuantizing, None, train_id, seq, None)
            model = dequantize(model)
            config.pop("quantization", None)

        # Save the merged model
        weights = dict(tree_flatten(model.parameters()))
        save_path = Path(save_path)
        save(
            save_path,
            model_path,
            weights,
            tokenizer,
            config,
            hf_repo=None,  
            donate_weights=False,
        )
        write_log(LevelEnum.INFO,LogEnum.MergeSuccess,f"{save_path}",train_id,seq,None)
        return {
            "status": "success",
            "message": f"Model successfully merged and saved to {save_path}"
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
        result = merge_model(
            train_id = args.train_id,
            base_model_path = args.base_path,
            adapter_path = args.adapter_path,
            save_path = args.save_path,
        )
        
        # 输出结果
        print(json.dumps(result))
        
        # 根据结果设置退出码
        if result.get('status') == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        error_result = {
            "status": "error",
            "message": f"Script execution failed: {str(e)}"
        }
        print(json.dumps(error_result))
        sys.exit(1)