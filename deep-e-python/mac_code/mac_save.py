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
