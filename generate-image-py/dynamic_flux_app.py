import torch
import base64
import io
import os
import uuid
import asyncio
import threading
import time
import warnings
import logging
from datetime import datetime, timedelta
from PIL import Image
from flask import Flask, request, jsonify
from diffusers.utils import load_image
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, PriorityQueue, Empty
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from enum import Enum
from pathlib import Path

# 导入自定义模型管理器
from model_manager import DynamicModelManager, ModelInfo

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 过滤警告
warnings.filterwarnings("ignore", message="The model may be quantized to fp4, but you are loading it with int4 precision.")

app = Flask(__name__)

# Hugging Face 认证信息
HF_USERNAME = ""
HF_TOKEN = ""

# 任务状态枚举
class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class GenerationTask:
    task_id: str
    task_type: str  # "text2image" or "image2image"
    model: str
    prompt: str
    batch_size: int
    aspect_ratio: str
    folder_uuid: str  # 文件夹UUID
    status: TaskStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    result_files: Optional[list] = None
    error_message: Optional[str] = None
    priority: int = 0  # 0 = highest priority
    
    # 图生图相关参数
    image: Optional[str] = None
    strength: Optional[float] = None
    
    # LoRA融合相关参数
    base_model_path: Optional[str] = None
    lora_weights_path: Optional[str] = None
    lora_scale: Optional[float] = None
    
    def __lt__(self, other):
        return self.priority < other.priority

# 全局变量
model_manager: Optional[DynamicModelManager] = None
task_queue = PriorityQueue()  # 任务队列
task_storage: Dict[str, GenerationTask] = {}  # 任务存储
executor = ThreadPoolExecutor(max_workers=1)  # 单线程执行器，确保一次只处理一个任务
cleanup_thread = None

# 默认图片存储路径
DEFAULT_OUTPUT_BASE = "./generated_images"

def init_model_manager():
    """初始化模型管理器"""
    global model_manager
    
    try:
        logger.info("初始化动态模型管理器...")
        model_manager = DynamicModelManager(
            config_path="models_config.yaml",
            hf_token=HF_TOKEN
        )
        logger.info("模型管理器初始化成功")
        
        # 显示可用模型
        models = model_manager.list_models()
        logger.info(f"可用模型数量: {len(models)}")
        for model_id, model_info in models.items():
            logger.info(f"  - {model_id}: {model_info.name} ({model_info.model_type})")
            
    except Exception as e:
        logger.error(f"模型管理器初始化失败: {e}")
        raise

def base64_to_image(base64_str):
    """将base64字符串转换为PIL图像"""
    try:
        # 移除data:image/...;base64,前缀（如果存在）
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Base64转图像失败: {e}")
        raise ValueError(f"无效的base64图像数据: {e}")

def get_dimensions_from_aspect_ratio(aspect_ratio, base_size=768):
    """根据宽高比获取图像尺寸，确保宽高都能被16整除"""
    def round_to_multiple_of_16(value):
        """将数值四舍五入到最接近的16的倍数"""
        return int(round(value / 16) * 16)
    
    aspect_ratios = {
        "1:1": (base_size, base_size),
        "16:9": (round_to_multiple_of_16(base_size * 16/9), base_size),
        "9:16": (base_size, round_to_multiple_of_16(base_size * 16/9)),
        "4:3": (round_to_multiple_of_16(base_size * 4/3), base_size),
        "3:4": (base_size, round_to_multiple_of_16(base_size * 4/3)),
        "21:9": (round_to_multiple_of_16(base_size * 21/9), base_size),
        "9:21": (base_size, round_to_multiple_of_16(base_size * 21/9))
    }
    return aspect_ratios.get(aspect_ratio, (base_size, base_size))

def save_images(images, folder_uuid):
    """保存生成的图像"""
    try:
        # 创建输出目录
        output_dir = Path(DEFAULT_OUTPUT_BASE) / folder_uuid
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for i, image in enumerate(images):
            filename = f"generated_{int(time.time())}_{i}.png"
            filepath = output_dir / filename
            
            # 保存图像
            image.save(filepath, "PNG")
            # 返回相对路径格式：folder_uuid/filename
            relative_path = f"{folder_uuid}/{filename}"
            saved_files.append(relative_path)
            
        logger.info(f"保存了 {len(saved_files)} 张图像到 {output_dir}")
        return saved_files
        
    except Exception as e:
        logger.error(f"保存图像失败: {e}")
        raise

def process_task(task: GenerationTask):
    """处理生成任务"""
    try:
        logger.info(f"开始处理任务: {task.task_id} ({task.task_type})")
        task.status = TaskStatus.PROCESSING
        
        # 获取模型配置信息
        model_config = model_manager.get_model_config(task.model)
        
        # 检查是否应该使用LoRA融合模式
        use_lora_fusion = False
        if model_config and model_config.get('type') == 'stable_diffusion':
            # 检查配置中是否有LoRA相关参数
            has_base_model = 'base_model_path' in model_config and model_config['base_model_path']
            has_lora_weights = 'lora_weights_path' in model_config and model_config['lora_weights_path']
            has_lora_scale = 'lora_scale' in model_config
            
            if has_base_model and has_lora_weights:
                use_lora_fusion = True
                # 从配置中获取LoRA参数
                task.base_model_path = model_config['base_model_path']
                task.lora_weights_path = model_config['lora_weights_path']
                task.lora_scale = model_config.get('lora_scale', 0.7)
                logger.info(f"检测到stable_diffusion模型配置中的LoRA参数，启用融合模式")
        
        # 如果用户直接指定了LoRA参数，也使用融合模式
        if task.base_model_path and task.lora_weights_path:
            use_lora_fusion = True
        
        if use_lora_fusion:
            # LoRA融合模式：直接使用指定的路径，不依赖model_manager
            pipeline = None
            model_info = None
            logger.info(f"使用LoRA融合模式: base_model={task.base_model_path}, lora_weights={task.lora_weights_path}")
            
            # 验证路径存在性
            if not os.path.exists(task.base_model_path):
                raise ValueError(f"基础模型路径不存在: {task.base_model_path}")
            if not os.path.exists(task.lora_weights_path):
                raise ValueError(f"LoRA权重路径不存在: {task.lora_weights_path}")
            
            try:
                # 重新加载基础模型（如果当前模型不是指定的基础模型）
                from diffusers import DiffusionPipeline
                logger.info(f"加载基础模型: {task.base_model_path}")
                pipeline = DiffusionPipeline.from_pretrained(
                    task.base_model_path,
                    torch_dtype=torch.float16,
                    device_map="balanced"
                )
                
                # 加载LoRA权重
                logger.info(f"加载LoRA权重: {task.lora_weights_path}")
                # 检查是否是safetensors文件
                if task.lora_weights_path.endswith('.safetensors'):
                    pipeline.load_lora_weights(os.path.dirname(task.lora_weights_path), 
                                             weight_name=os.path.basename(task.lora_weights_path))
                else:
                    pipeline.load_lora_weights(task.lora_weights_path)
                
                # 融合LoRA参数
                lora_scale = task.lora_scale if task.lora_scale is not None else 0.7
                logger.info(f"融合LoRA权重，缩放因子: {lora_scale}")
                pipeline.fuse_lora(lora_scale=lora_scale)
                
                logger.info("LoRA融合完成")
                
            except Exception as e:
                logger.error(f"LoRA融合失败: {e}")
                raise ValueError(f"LoRA融合失败: {e}")
        else:
            # 常规模式：使用model_manager加载模型
            logger.info(f"使用常规模式加载模型: {task.model}")
            pipeline, model_info = model_manager.get_model(task.model)
        
        # 获取图像尺寸
        width, height = get_dimensions_from_aspect_ratio(task.aspect_ratio)
        
        # 针对flux.1-schnell优化推理步数
        is_flux_schnell = 'schnell' in task.model.lower()
        num_inference_steps = 4 if is_flux_schnell else 20  # flux.1-schnell使用更少步数
        guidance_scale = 0.0 if is_flux_schnell else 7.5  # flux.1-schnell不需要guidance
        
        generation_kwargs = {
            "prompt": task.prompt,
            "width": width,
            "height": height,
            "num_images_per_prompt": task.batch_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": torch.Generator(device="cuda").manual_seed(int(time.time()))
        }
        
        logger.info(f"使用推理步数: {num_inference_steps}, 引导尺度: {guidance_scale}")
        
        # 根据任务类型生成图像
        if task.task_type == "text2image":
            logger.info(f"执行文生图任务: {task.prompt[:50]}...")
            
            # 检查模型是否支持文生图（LoRA融合模式跳过检查）
            if model_info and not model_info.config.get('supports_text2img', True):
                raise ValueError(f"模型 {task.model} 不支持文生图")
                
            result = pipeline(**generation_kwargs)
            
        elif task.task_type == "image2image":
            logger.info(f"执行图生图任务: {task.prompt[:50]}...")
            
            # 检查模型是否支持图生图（LoRA融合模式跳过检查）
            if model_info and not model_info.config.get('supports_img2img', False):
                raise ValueError(f"模型 {task.model} 不支持图生图")
                
            if not task.image:
                raise ValueError("图生图任务缺少输入图像")
                
            # 转换输入图像
            input_image = base64_to_image(task.image)
            
            generation_kwargs.update({
                "image": input_image
            })
            
            # 只有支持strength参数的pipeline才添加strength参数
            # FluxKontextPipeline不支持strength参数
            pipeline_class = model_info.config.get('pipeline_class', '')
            if pipeline_class != 'FluxKontextPipeline':
                generation_kwargs["strength"] = task.strength or 0.8
            
            result = pipeline(**generation_kwargs)
            
        else:
            raise ValueError(f"不支持的任务类型: {task.task_type}")
        
        # 保存生成的图像
        images = result.images
        saved_files = save_images(images, task.folder_uuid)
        
        # LoRA融合后的清理工作
        if task.base_model_path and task.lora_weights_path:
            logger.info("LoRA融合任务完成，开始清理显存...")
            try:
                # 删除融合后的pipeline引用
                del pipeline
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    for _ in range(3):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                # 强制卸载model_manager中的当前模型（如果有）
                if model_manager and model_manager.current_pipeline:
                    model_manager._unload_current_model()
                
                logger.info("LoRA融合任务显存清理完成")
            except Exception as cleanup_error:
                logger.warning(f"LoRA融合后清理显存时出现警告: {cleanup_error}")
        
        # 更新任务状态
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result_files = saved_files
        
        logger.info(f"任务完成: {task.task_id}, 生成了 {len(saved_files)} 张图像")
        
    except Exception as e:
        logger.error(f"任务处理失败: {task.task_id}, 错误: {e}")
        
        # LoRA融合任务失败时也需要清理显存
        if task.base_model_path and task.lora_weights_path:
            logger.info("LoRA融合任务失败，开始清理显存...")
            try:
                # 尝试删除pipeline引用（如果存在）
                if 'pipeline' in locals():
                    del pipeline
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    for _ in range(3):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                # 强制卸载model_manager中的当前模型（如果有）
                if model_manager and model_manager.current_pipeline:
                    model_manager._unload_current_model()
                
                logger.info("LoRA融合任务失败后显存清理完成")
            except Exception as cleanup_error:
                logger.warning(f"LoRA融合任务失败后清理显存时出现警告: {cleanup_error}")
        
        task.status = TaskStatus.FAILED
        task.error_message = str(e)
        task.completed_at = datetime.now()

def task_worker():
    """任务工作线程"""
    logger.info("任务工作线程启动")
    
    while True:
        try:
            # 从队列获取任务（阻塞等待）
            task = task_queue.get(timeout=1)
            
            # 处理任务
            process_task(task)
            
            # 标记任务完成
            task_queue.task_done()
            
        except Empty:
            # 队列为空，继续等待
            continue
        except Exception as e:
            logger.error(f"任务工作线程出错: {e}")
            time.sleep(1)

def cleanup_old_files():
    """清理旧文件的线程"""
    logger.info("文件清理线程启动")
    
    while True:
        try:
            # 每小时清理一次
            time.sleep(3600)
            
            # 清理7天前的文件
            cutoff_time = datetime.now() - timedelta(days=7)
            base_path = Path(DEFAULT_OUTPUT_BASE)
            
            if not base_path.exists():
                continue
                
            deleted_count = 0
            for folder in base_path.iterdir():
                if folder.is_dir():
                    try:
                        # 检查文件夹的修改时间
                        folder_time = datetime.fromtimestamp(folder.stat().st_mtime)
                        if folder_time < cutoff_time:
                            # 删除整个文件夹
                            import shutil
                            shutil.rmtree(folder)
                            deleted_count += 1
                            logger.info(f"删除过期文件夹: {folder}")
                    except Exception as e:
                        logger.warning(f"删除文件夹失败 {folder}: {e}")
                        
            if deleted_count > 0:
                logger.info(f"清理完成，删除了 {deleted_count} 个过期文件夹")
                
        except Exception as e:
            logger.error(f"文件清理线程出错: {e}")

# 启动工作线程
worker_thread = threading.Thread(target=task_worker, daemon=True)
worker_thread.start()

cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

def check_output_directory():
    """检查输出目录权限"""
    try:
        output_path = Path(DEFAULT_OUTPUT_BASE)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 测试写入权限
        test_file = output_path / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        
        logger.info(f"输出目录检查通过: {DEFAULT_OUTPUT_BASE}")
        return True
        
    except Exception as e:
        logger.error(f"输出目录权限检查失败: {e}")
        return False

# ==================== API 路由 ====================

@app.route('/text2image', methods=['POST'])
def text_to_image():
    """文生图API"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        if 'prompt' not in data:
            return jsonify({"error": "缺少必需参数: prompt"}), 400
        
        # 支持 model 和 model_name 两种参数名
        model_name = data.get('model') or data.get('model_name')
        if not model_name:
            return jsonify({"error": "缺少必需参数: model 或 model_name"}), 400
        
        # 验证模型是否存在
        if not model_manager.get_model_info(model_name):
            return jsonify({"error": f"未知的模型: {model_name}"}), 400
        
        # 创建任务
        task_id = str(uuid.uuid4())
        folder_uuid = str(uuid.uuid4())
        
        # 针对flux.1-schnell优化：降低默认batch_size以减少显存占用
        default_batch_size = 1 if model_name == 'flux.1-schnell' else 1
        
        task = GenerationTask(
            task_id=task_id,
            task_type="text2image",
            model=model_name,
            prompt=data['prompt'],
            batch_size=data.get('batch_size', data.get('batchSize', default_batch_size)),
            aspect_ratio=data.get('aspect_ratio', data.get('aspectRatio', '1:1')),
            folder_uuid=folder_uuid,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            priority=data.get('priority', 0),
            # LoRA融合相关参数
            base_model_path=data.get('base_model_path'),
            lora_weights_path=data.get('lora_weights_path'),
            lora_scale=data.get('lora_scale', 0.7)
        )
        
        # 存储任务
        task_storage[task_id] = task
        
        # 添加到队列
        task_queue.put(task)
        
        logger.info(f"创建文生图任务: {task_id}")
        
        return jsonify({
            "task_id": task_id,
            "status": "pending",
            "message": "任务已创建，正在处理中"
        })
        
    except Exception as e:
        logger.error(f"文生图API错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/image2image', methods=['POST'])
def image_to_image():
    """图生图API"""
    try:
        data = request.get_json()
        
        # 验证必需参数
        if 'prompt' not in data:
            return jsonify({"error": "缺少必需参数: prompt"}), 400
        
        # 支持 image 和 base64_image 两种参数名
        image_data = data.get('image') or data.get('base64_image')
        if not image_data:
            return jsonify({"error": "缺少必需参数: image 或 base64_image"}), 400
        
        # 支持 model 和 model_name 两种参数名
        model_name = data.get('model') or data.get('model_name')
        if not model_name:
            return jsonify({"error": "缺少必需参数: model 或 model_name"}), 400
        
        # 验证模型是否存在
        model_info = model_manager.get_model_info(model_name)
        if not model_info:
            return jsonify({"error": f"未知的模型: {model_name}"}), 400
            
        # 验证模型是否支持图生图
        if not model_info.config.get('supports_img2img', False):
            return jsonify({"error": f"模型 {model_name} 不支持图生图"}), 400
        
        # 创建任务
        task_id = str(uuid.uuid4())
        folder_uuid = str(uuid.uuid4())
        
        # 针对flux.1-schnell优化：降低默认batch_size以减少显存占用
        default_batch_size = 1 if model_name == 'flux.1-schnell' else 1
        
        task = GenerationTask(
            task_id=task_id,
            task_type="image2image",
            model=model_name,
            prompt=data['prompt'],
            batch_size=data.get('batch_size', data.get('batchSize', default_batch_size)),
            aspect_ratio=data.get('aspect_ratio', data.get('aspectRatio', '1:1')),
            folder_uuid=folder_uuid,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            priority=data.get('priority', 0),
            image=image_data,
            strength=data.get('strength', 0.8),
            # LoRA融合相关参数
            base_model_path=data.get('base_model_path'),
            lora_weights_path=data.get('lora_weights_path'),
            lora_scale=data.get('lora_scale', 0.7)
        )
        
        # 存储任务
        task_storage[task_id] = task
        
        # 添加到队列
        task_queue.put(task)
        
        logger.info(f"创建图生图任务: {task_id}")
        
        return jsonify({
            "task_id": task_id,
            "status": "pending",
            "message": "任务已创建，正在处理中"
        })
        
    except Exception as e:
        logger.error(f"图生图API错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """获取任务状态"""
    try:
        if task_id not in task_storage:
            return jsonify({"error": "任务不存在"}), 404
        
        task = task_storage[task_id]
        
        response = {
            "task_id": task.task_id,
            "status": task.status.value,
            "task_type": task.task_type,
            "model": task.model,
            "prompt": task.prompt,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None
        }
        
        if task.status == TaskStatus.COMPLETED:
            response["result_files"] = task.result_files
        elif task.status == TaskStatus.FAILED:
            response["error_message"] = task.error_message
            
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"获取任务状态错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/tasks', methods=['GET'])
def list_tasks():
    """列出所有任务"""
    try:
        # 获取查询参数
        status_filter = request.args.get('status')
        limit = int(request.args.get('limit', 50))
        
        tasks = []
        for task in list(task_storage.values()):
            if status_filter and task.status.value != status_filter:
                continue
                
            task_info = {
                "task_id": task.task_id,
                "status": task.status.value,
                "task_type": task.task_type,
                "model": task.model,
                "prompt": task.prompt[:100] + "..." if len(task.prompt) > 100 else task.prompt,
                "created_at": task.created_at.isoformat(),
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
            
            if task.status == TaskStatus.FAILED:
                task_info["error_message"] = task.error_message
                
            tasks.append(task_info)
        
        # 按创建时间排序（最新的在前）
        tasks.sort(key=lambda x: x['created_at'], reverse=True)
        
        # 限制返回数量
        tasks = tasks[:limit]
        
        return jsonify({
            "tasks": tasks,
            "total": len(task_storage),
            "filtered": len(tasks)
        })
        
    except Exception as e:
        logger.error(f"列出任务错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,
                "gpu_memory_cached": torch.cuda.memory_reserved(0) / 1024**3,
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        else:
            gpu_info = {"gpu_available": False}
        
        model_status = model_manager.get_current_model_status() if model_manager else {}
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "worker_thread_alive": worker_thread.is_alive(),
            "cleanup_thread_alive": cleanup_thread.is_alive(),
            "queue_size": task_queue.qsize(),
            "total_tasks": len(task_storage),
            "model_manager_initialized": model_manager is not None,
            "current_model": model_status,
            "gpu_info": gpu_info
        })
        
    except Exception as e:
        logger.error(f"健康检查错误: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== 模型管理API ====================

@app.route('/models', methods=['GET'])
def list_models():
    """列出所有可用模型"""
    try:
        if not model_manager:
            return jsonify({"error": "模型管理器未初始化"}), 500
            
        models = model_manager.list_models()
        model_list = []
        
        for model_id, model_info in models.items():
            model_data = {
                "id": model_id,
                "name": model_info.name,
                "type": model_info.model_type,
                "pipeline_class": model_info.pipeline_class,
                "memory_usage": model_info.memory_usage,
                "is_loaded": model_info.is_loaded,
                "last_used": model_info.last_used.isoformat() if model_info.last_used else None,
                "supports_text2img": model_info.config.get('supports_text2img', True),
                "supports_img2img": model_info.config.get('supports_img2img', False),
                "description": model_info.config.get('description', '')
            }
            model_list.append(model_data)
            
        return jsonify({
            "models": model_list,
            "total": len(model_list)
        })
        
    except Exception as e:
        logger.error(f"列出模型错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/<model_id>', methods=['GET'])
def get_model_info(model_id):
    """获取指定模型信息"""
    try:
        if not model_manager:
            return jsonify({"error": "模型管理器未初始化"}), 500
            
        model_info = model_manager.get_model_info(model_id)
        if not model_info:
            return jsonify({"error": f"模型不存在: {model_id}"}), 404
            
        return jsonify({
            "id": model_id,
            "name": model_info.name,
            "type": model_info.model_type,
            "pipeline_class": model_info.pipeline_class,
            "memory_usage": model_info.memory_usage,
            "is_loaded": model_info.is_loaded,
            "last_used": model_info.last_used.isoformat() if model_info.last_used else None,
            "config": model_info.config
        })
        
    except Exception as e:
        logger.error(f"获取模型信息错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/<model_id>', methods=['PUT'])
def update_model_config(model_id):
    """更新模型配置"""
    try:
        if not model_manager:
            return jsonify({"error": "模型管理器未初始化"}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "请提供配置数据"}), 400
            
        model_manager.update_model(model_id, data)
        
        return jsonify({
            "message": f"模型配置已更新: {model_id}",
            "model_id": model_id
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"更新模型配置错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/<model_id>', methods=['DELETE'])
def delete_model_config(model_id):
    """删除模型配置"""
    try:
        if not model_manager:
            return jsonify({"error": "模型管理器未初始化"}), 500
            
        model_manager.remove_model(model_id)
        
        return jsonify({
            "message": f"模型配置已删除: {model_id}",
            "model_id": model_id
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"删除模型配置错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['POST'])
def add_model_config():
    """添加新模型配置"""
    try:
        if not model_manager:
            return jsonify({"error": "模型管理器未初始化"}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "请提供模型配置数据"}), 400
            
        # 验证必需字段
        required_fields = ['model_id', 'name', 'type', 'pipeline_class']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"缺少必需字段: {field}"}), 400
                
        model_id = data.pop('model_id')
        model_manager.add_model(model_id, data)
        
        return jsonify({
            "message": f"模型配置已添加: {model_id}",
            "model_id": model_id
        }), 201
        
    except Exception as e:
        logger.error(f"添加模型配置错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/current', methods=['GET'])
def get_current_model():
    """获取当前加载的模型状态"""
    try:
        if not model_manager:
            return jsonify({"error": "模型管理器未初始化"}), 500
            
        status = model_manager.get_current_model_status()
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"获取当前模型状态错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/models/unload', methods=['POST'])
def unload_current_model():
    """手动卸载当前模型"""
    try:
        if not model_manager:
            return jsonify({"error": "模型管理器未初始化"}), 500
            
        with model_manager.model_lock:
            if model_manager.current_model:
                current_model = model_manager.current_model
                model_manager._unload_current_model()
                return jsonify({
                    "message": f"模型已卸载: {current_model}"
                })
            else:
                return jsonify({
                    "message": "当前没有加载的模型"
                })
                
    except Exception as e:
        logger.error(f"卸载模型错误: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/queue/clear', methods=['POST'])
def clear_queue():
    """清空任务队列"""
    try:
        # 清空队列
        while not task_queue.empty():
            try:
                task = task_queue.get_nowait()
                if task.task_id in task_storage:
                    task_storage[task.task_id].status = TaskStatus.FAILED
                    task_storage[task.task_id].error_message = "任务被取消"
                    task_storage[task.task_id].completed_at = datetime.now()
            except Empty:
                break
                
        return jsonify({
            "message": "任务队列已清空",
            "remaining_tasks": task_queue.qsize()
        })
        
    except Exception as e:
        logger.error(f"清空队列错误: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== 应用启动 ====================

if __name__ == '__main__':
    # 检查CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA不可用！此应用需要GPU运行。")
        exit(1)
        
    logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 检查输出目录
    if not check_output_directory():
        logger.warning("输出目录权限检查失败，程序仍会启动，但保存图片时可能会出错")
    
    # 初始化模型管理器
    init_model_manager()
    
    logger.info("启动动态模型管理服务...")
    logger.info(f"工作线程状态: {worker_thread.is_alive()}")
    logger.info(f"清理线程状态: {cleanup_thread.is_alive()}")
    
    # 启动Flask应用
    try:
        from waitress import serve
        logger.info("使用Waitress生产服务器启动...")
        serve(app, host='0.0.0.0', port=5050, threads=4)
    except ImportError:
        logger.info("Waitress未安装，使用Flask开发服务器...")
        logger.info("建议安装: pip install waitress")
        app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
    finally:
        # 清理资源
        if model_manager:
            model_manager.shutdown()
        logger.info("应用已关闭")