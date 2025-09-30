import torch
import yaml
import threading
import time
import logging
import gc
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

# 导入所需的管道类
from diffusers import FluxPipeline, FluxKontextPipeline
try:
    from diffusers import StableDiffusion3Pipeline
except ImportError:
    StableDiffusion3Pipeline = None
    
from nunchaku import NunchakuFluxTransformer2dModel
from huggingface_hub import login

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """模型信息数据类"""
    name: str
    model_type: str
    pipeline_class: str
    config: Dict[str, Any]
    last_used: Optional[datetime] = None
    is_loaded: bool = False
    memory_usage: str = "medium"

class ModelLoader(ABC):
    """模型加载器抽象基类"""
    
    @abstractmethod
    def load_model(self, config: Dict[str, Any]) -> Any:
        """加载模型"""
        pass
    
    @abstractmethod
    def unload_model(self, pipeline: Any) -> None:
        """卸载模型"""
        pass

class FluxModelLoader(ModelLoader):
    """FLUX模型加载器"""
    
    def load_model(self, config: Dict[str, Any]) -> Any:
        """加载FLUX模型"""
        logger.info(f"加载FLUX模型: {config['name']}")
        
        # 清理GPU缓存（参考flux_release_edit_0806.py）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 确定模型路径
        model_path = self._resolve_model_path(config)
        
        # 根据管道类型创建管道
        pipeline_class_name = config.get('pipeline_class', 'FluxPipeline')
        if pipeline_class_name == 'FluxKontextPipeline':
            pipeline_class = FluxKontextPipeline
        else:
            pipeline_class = FluxPipeline
            
        try:
            # 针对flux.1-schnell的特殊内存优化
            model_name = config.get('name', '').lower()
            is_flux_schnell = 'schnell' in model_name
            
            # 加载管道时的内存优化参数
            load_kwargs = {
                "torch_dtype": getattr(torch, config.get('torch_dtype', 'bfloat16')),
                "low_cpu_mem_usage": config.get('low_cpu_mem_usage', True),
                "use_safetensors": True,
                "local_files_only": True
            }
            
            # flux.1-schnell额外的加载优化
            if is_flux_schnell:
                # 移除fp16变体设置，因为该变体不存在
                logger.info("为flux.1-schnell启用内存优化加载")
            
            # 加载管道
            pipeline = pipeline_class.from_pretrained(model_path, **load_kwargs)
            
            # 移动到GPU前先启用部分优化
            if is_flux_schnell:
                # 在移动到GPU前启用CPU offload相关优化
                if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                    pipeline.enable_sequential_cpu_offload()
                    logger.info("在GPU移动前启用顺序CPU offload")
                elif hasattr(pipeline, 'enable_model_cpu_offload'):
                    pipeline.enable_model_cpu_offload()
                    logger.info("在GPU移动前启用模型CPU offload")
            else:
                # 非flux.1-schnell模型正常移动到GPU
                pipeline = pipeline.to("cuda")
            
            # 启用优化
            self._enable_optimizations(pipeline, config)
            
            # 记录GPU内存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"模型加载完成 - GPU内存使用: 已分配 {allocated:.2f}GB, 已缓存 {cached:.2f}GB")
            
            return pipeline
            
        except Exception as e:
            # 加载失败时清理GPU内存
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.error(f"模型加载失败: {e}")
            raise
    
    def unload_model(self, pipeline: Any) -> None:
        """卸载FLUX模型"""
        if pipeline is None:
            return
            
        logger.info("卸载FLUX模型")
        
        # 清理各个组件
        components = ['unet', 'transformer', 'vae', 'text_encoder', 'text_encoder_2']
        for component in components:
            if hasattr(pipeline, component):
                setattr(pipeline, component, None)
        
        del pipeline
    
    def _resolve_model_path(self, config: Dict[str, Any]) -> str:
        """解析模型路径"""
        # 优先使用本地路径
        local_path = config.get('local_path')
        if local_path and Path(local_path).exists():
            return local_path
            
        # 使用HuggingFace路径
        hf_path = config.get('huggingface_path')
        if hf_path:
            return hf_path
            
        raise FileNotFoundError(f"模型路径未找到: {config['name']}")
    
    def _enable_optimizations(self, pipeline: Any, config: Dict[str, Any]) -> None:
        """启用模型优化"""
        try:
            # 确保所有组件使用一致的数据类型（参考flux_release_edit_0806.py）
            target_dtype = torch.bfloat16
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                pipeline.vae = pipeline.vae.to(target_dtype)
            if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                pipeline.text_encoder = pipeline.text_encoder.to(target_dtype)
            if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
                pipeline.text_encoder_2 = pipeline.text_encoder_2.to(target_dtype)
            
            # 启用VAE优化
            if hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()
            if hasattr(pipeline, 'enable_vae_tiling'):
                pipeline.enable_vae_tiling()
            
            # 针对flux.1-schnell的特殊优化
            model_name = config.get('name', '').lower()
            is_flux_schnell = 'schnell' in model_name
            
            # 启用注意力优化 - flux.1-schnell使用更小的slice以节省内存
            if hasattr(pipeline, 'enable_attention_slicing'):
                attention_slice = "auto" if is_flux_schnell else 1
                pipeline.enable_attention_slicing(attention_slice)
                logger.info(f"启用注意力切片: {attention_slice}")
            
            # 根据配置启用CPU offload - flux.1-schnell强制使用顺序offload
            enable_cpu_offload = self.settings.get('enable_cpu_offload', True)
            enable_sequential_offload = self.settings.get('enable_sequential_cpu_offload', False) or is_flux_schnell
            
            if enable_sequential_offload and hasattr(pipeline, 'enable_sequential_cpu_offload'):
                pipeline.enable_sequential_cpu_offload()
                logger.info("启用了顺序CPU offload")
            elif enable_cpu_offload and hasattr(pipeline, 'enable_model_cpu_offload'):
                pipeline.enable_model_cpu_offload()
                logger.info("启用了CPU offload")
            
            # flux.1-schnell额外的内存优化
            if is_flux_schnell:
                # 启用更积极的内存清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("为flux.1-schnell执行额外的GPU内存清理")
                
                # 尝试启用低内存模式
                if hasattr(pipeline, 'enable_freeu'):
                    try:
                        pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
                        logger.info("为flux.1-schnell启用FreeU优化")
                    except Exception:
                        pass
            
            # 尝试启用xformers
            try:
                if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("启用了xformers内存优化")
            except Exception:
                logger.info("xformers不可用，使用默认注意力机制")
                
        except Exception as e:
            logger.warning(f"启用优化时出错: {e}")

class NunchakuModelLoader(ModelLoader):
    """Nunchaku优化模型加载器"""
    
    def load_model(self, config: Dict[str, Any]) -> Any:
        """加载Nunchaku优化模型"""
        logger.info(f"加载Nunchaku模型: {config['name']}")
        
        # 清理GPU缓存（参考flux_release_edit_0806.py）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        try:
            # 加载量化的transformer
            transformer_path = config['local_path']
            if not Path(transformer_path).exists():
                raise FileNotFoundError(f"Nunchaku模型文件未找到: {transformer_path}")
                
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                transformer_path,
                torch_dtype=getattr(torch, config.get('torch_dtype', 'bfloat16'))
            )
            
            # 使用基础模型和量化transformer创建管道
            base_model = config['base_model']
            pipeline = FluxKontextPipeline.from_pretrained(
                base_model,
                transformer=transformer,
                torch_dtype=getattr(torch, config.get('torch_dtype', 'bfloat16')),
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            
            # 移动到GPU
            pipeline = pipeline.to("cuda")
            
            # 启用优化
            self._enable_optimizations(pipeline)
            
            # 记录GPU内存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Nunchaku模型加载完成 - GPU内存使用: 已分配 {allocated:.2f}GB, 已缓存 {cached:.2f}GB")
            
            return pipeline
            
        except Exception as e:
            # 加载失败时清理GPU内存
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.error(f"Nunchaku模型加载失败: {e}")
            raise
    
    def unload_model(self, pipeline: Any) -> None:
        """卸载Nunchaku模型"""
        if pipeline is None:
            return
            
        logger.info("卸载Nunchaku模型")
        
        # 清理各个组件
        components = ['transformer', 'vae', 'text_encoder', 'text_encoder_2']
        for component in components:
            if hasattr(pipeline, component):
                setattr(pipeline, component, None)
        
        del pipeline
    
    def _enable_optimizations(self, pipeline: Any) -> None:
        """启用模型优化"""
        try:
            pipeline.enable_vae_slicing()
            pipeline.enable_vae_tiling()
            pipeline.enable_attention_slicing(1)
            pipeline.enable_model_cpu_offload()
            
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("启用了xformers内存优化")
            except Exception:
                logger.info("xformers不可用，使用默认注意力机制")
                
        except Exception as e:
            logger.warning(f"启用优化时出错: {e}")

class StableDiffusionModelLoader(ModelLoader):
    """Stable Diffusion模型加载器"""
    
    def load_model(self, config: Dict[str, Any]) -> Any:
        """加载Stable Diffusion模型"""
        if StableDiffusion3Pipeline is None:
            raise ImportError("StableDiffusion3Pipeline不可用，请更新diffusers库")
            
        logger.info(f"加载Stable Diffusion模型: {config['name']}")
        
        # 清理GPU缓存（参考flux_release_edit_0806.py）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        try:
            model_path = self._resolve_model_path(config)
            
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, config.get('torch_dtype', 'bfloat16')),
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=True
            )
            
            pipeline = pipeline.to("cuda")
            self._enable_optimizations(pipeline)
            
            # 记录GPU内存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Stable Diffusion模型加载完成 - GPU内存使用: 已分配 {allocated:.2f}GB, 已缓存 {cached:.2f}GB")
            
            return pipeline
            
        except Exception as e:
            # 加载失败时清理GPU内存
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.error(f"Stable Diffusion模型加载失败: {e}")
            raise
    
    def unload_model(self, pipeline: Any) -> None:
        """卸载Stable Diffusion模型"""
        if pipeline is None:
            return
            
        logger.info("卸载Stable Diffusion模型")
        
        components = ['transformer', 'vae', 'text_encoder', 'text_encoder_2', 'text_encoder_3']
        for component in components:
            if hasattr(pipeline, component):
                setattr(pipeline, component, None)
        
        del pipeline
    
    def _resolve_model_path(self, config: Dict[str, Any]) -> str:
        """解析模型路径"""
        local_path = config.get('local_path')
        if local_path and Path(local_path).exists():
            return local_path
            
        hf_path = config.get('huggingface_path')
        if hf_path:
            return hf_path
            
        raise FileNotFoundError(f"模型路径未找到: {config['name']}")
    
    def _enable_optimizations(self, pipeline: Any) -> None:
        """启用模型优化"""
        try:
            if hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()
            if hasattr(pipeline, 'enable_vae_tiling'):
                pipeline.enable_vae_tiling()
            if hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing(1)
            if hasattr(pipeline, 'enable_model_cpu_offload'):
                pipeline.enable_model_cpu_offload()
                
        except Exception as e:
            logger.warning(f"启用优化时出错: {e}")

class GGUFModelLoader(ModelLoader):
    """GGUF格式模型加载器"""
    
    def load_model(self, config: Dict[str, Any]) -> Any:
        """加载GGUF模型"""
        logger.info(f"加载GGUF模型: {config['name']}")
        
        # 清理GPU缓存（参考flux_release_edit_0806.py）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        try:
            # GGUF模型通常需要特殊处理，这里简化为使用FluxKontextPipeline
            model_path = self._resolve_model_path(config)
            
            pipeline = FluxKontextPipeline.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, config.get('torch_dtype', 'bfloat16')),
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            
            pipeline = pipeline.to("cuda")
            self._enable_optimizations(pipeline)
            
            # 记录GPU内存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GGUF模型加载完成 - GPU内存使用: 已分配 {allocated:.2f}GB, 已缓存 {cached:.2f}GB")
            
            return pipeline
            
        except Exception as e:
            # 加载失败时清理GPU内存
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.error(f"GGUF模型加载失败: {e}")
            raise
    
    def unload_model(self, pipeline: Any) -> None:
        """卸载GGUF模型"""
        if pipeline is None:
            return
            
        logger.info("卸载GGUF模型")
        
        components = ['transformer', 'vae', 'text_encoder', 'text_encoder_2']
        for component in components:
            if hasattr(pipeline, component):
                setattr(pipeline, component, None)
        
        del pipeline
    
    def _resolve_model_path(self, config: Dict[str, Any]) -> str:
        """解析模型路径"""
        local_path = config.get('local_path')
        if local_path and Path(local_path).exists():
            return local_path
            
        hf_path = config.get('huggingface_path')
        if hf_path:
            return hf_path
            
        raise FileNotFoundError(f"模型路径未找到: {config['name']}")
    
    def _enable_optimizations(self, pipeline: Any) -> None:
        """启用模型优化"""
        try:
            pipeline.enable_vae_slicing()
            pipeline.enable_vae_tiling()
            pipeline.enable_attention_slicing(1)
            pipeline.enable_model_cpu_offload()
            
        except Exception as e:
            logger.warning(f"启用优化时出错: {e}")

class DynamicModelManager:
    """动态模型管理器"""
    
    def __init__(self, config_path: str = "models_config.yaml", hf_token: str = None):
        self.config_path = Path(config_path)
        self.hf_token = hf_token
        self.models_config: Dict[str, ModelInfo] = {}
        self.settings: Dict[str, Any] = {}
        self.current_model: Optional[str] = None
        self.current_pipeline: Optional[Any] = None
        self.model_lock = threading.RLock()  # 使用可重入锁
        self.cleanup_thread: Optional[threading.Thread] = None
        self.shutdown_flag = threading.Event()
        
        # 模型加载器映射
        self.loaders = {
            'flux': FluxModelLoader(),
            'flux_nunchaku': NunchakuModelLoader(),
            'stable_diffusion': StableDiffusionModelLoader(),
            'gguf': GGUFModelLoader()
        }
        
        # 加载配置
        self.load_config()
        
        # 设置HuggingFace认证
        if self.hf_token:
            try:
                login(token=self.hf_token)
                logger.info("HuggingFace认证成功")
            except Exception as e:
                logger.warning(f"HuggingFace认证失败: {e}")
                logger.warning("将在没有认证的情况下继续运行，可能无法访问私有模型")
            
        # 启动清理线程
        self.start_cleanup_thread()
    
    def load_config(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        self.settings = config.get('settings', {})
        
        # 解析模型配置
        models = config.get('models', {})
        for model_id, model_config in models.items():
            self.models_config[model_id] = ModelInfo(
                name=model_config['name'],
                model_type=model_config['type'],
                pipeline_class=model_config['pipeline_class'],
                config=model_config,
                memory_usage=model_config.get('memory_usage', 'medium')
            )
            
        logger.info(f"加载了 {len(self.models_config)} 个模型配置")
    
    def save_config(self) -> None:
        """保存配置文件"""
        config = {
            'models': {},
            'settings': self.settings
        }
        
        for model_id, model_info in self.models_config.items():
            config['models'][model_id] = model_info.config
            
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        logger.info("配置文件已保存")
    
    def get_model(self, model_id: str) -> Tuple[Any, ModelInfo]:
        """获取模型，如果未加载则动态加载"""
        with self.model_lock:
            if model_id not in self.models_config:
                raise ValueError(f"未知的模型ID: {model_id}")
                
            model_info = self.models_config[model_id]
            
            # 如果请求的模型已经是当前模型，直接返回
            if self.current_model == model_id and self.current_pipeline is not None:
                model_info.last_used = datetime.now()
                logger.info(f"使用已加载的模型: {model_id}")
                return self.current_pipeline, model_info
            
            # 需要切换模型
            logger.info(f"切换模型: {self.current_model} -> {model_id}")
            
            # 卸载当前模型
            self._unload_current_model()
            
            # 加载新模型
            pipeline = self._load_model(model_id, model_info)
            
            # 更新状态
            self.current_model = model_id
            self.current_pipeline = pipeline
            model_info.is_loaded = True
            model_info.last_used = datetime.now()
            
            # 清理GPU缓存
            self._cleanup_gpu_memory()
            
            logger.info(f"模型切换完成: {model_id}")
            return pipeline, model_info
    
    def _load_model(self, model_id: str, model_info: ModelInfo) -> Any:
        """加载指定模型"""
        logger.info(f"开始加载模型: {model_id}")
        start_time = time.time()
        
        # 获取对应的加载器
        loader = self.loaders.get(model_info.model_type)
        if loader is None:
            raise ValueError(f"不支持的模型类型: {model_info.model_type}")
            
        try:
            pipeline = loader.load_model(model_info.config)
            load_time = time.time() - start_time
            logger.info(f"模型加载完成: {model_id}，耗时: {load_time:.2f}秒")
            
            # 显示GPU内存使用情况
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"GPU内存使用 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB")
                
            return pipeline
            
        except Exception as e:
            logger.error(f"模型加载失败: {model_id}, 错误: {e}")
            raise
    
    def _unload_current_model(self) -> None:
        """卸载当前模型"""
        if self.current_pipeline is None:
            return
            
        logger.info(f"卸载当前模型: {self.current_model}")
        
        try:
            # 获取当前模型信息
            if self.current_model and self.current_model in self.models_config:
                model_info = self.models_config[self.current_model]
                loader = self.loaders.get(model_info.model_type)
                if loader:
                    loader.unload_model(self.current_pipeline)
                    model_info.is_loaded = False
            
            self.current_pipeline = None
            self.current_model = None
            
            # 强制垃圾回收
            gc.collect()
            
            # 清理GPU缓存
            self._cleanup_gpu_memory()
            
            logger.info("模型卸载完成")
            
        except Exception as e:
            logger.error(f"模型卸载时出错: {e}")
    
    def _cleanup_gpu_memory(self) -> None:
        """清理GPU内存"""
        if torch.cuda.is_available():
            # 强制垃圾回收
            gc.collect()
            
            # 多次清理GPU缓存
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 记录清理后的内存状态
            memory_info = self._get_gpu_memory_info()
            logger.info(f"GPU内存清理完成 - 已分配: {memory_info['allocated']:.2f}GB, 已缓存: {memory_info['cached']:.2f}GB")
    
    def start_cleanup_thread(self) -> None:
        """启动清理线程"""
        if self.cleanup_thread is not None and self.cleanup_thread.is_alive():
            return
            
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="ModelCleanupThread"
        )
        self.cleanup_thread.start()
        logger.info("模型清理线程已启动")
    
    def _cleanup_worker(self) -> None:
        """清理工作线程"""
        timeout_minutes = self.settings.get('auto_unload_timeout', 30)
        check_interval = 60  # 每分钟检查一次
        
        while not self.shutdown_flag.wait(check_interval):
            try:
                with self.model_lock:
                    if self.current_model and self.current_pipeline:
                        model_info = self.models_config.get(self.current_model)
                        if model_info and model_info.last_used:
                            idle_time = datetime.now() - model_info.last_used
                            if idle_time > timedelta(minutes=timeout_minutes):
                                logger.info(f"模型空闲超时，自动卸载: {self.current_model}")
                                self._unload_current_model()
                                
            except Exception as e:
                logger.error(f"清理线程出错: {e}")
    
    def shutdown(self) -> None:
        """关闭管理器"""
        logger.info("关闭模型管理器")
        
        # 停止清理线程
        self.shutdown_flag.set()
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
            
        # 卸载当前模型
        with self.model_lock:
            self._unload_current_model()
            
        logger.info("模型管理器已关闭")
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return self.models_config.get(model_id)
    
    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型配置"""
        model_info = self.models_config.get(model_id)
        return model_info.config if model_info else None
    
    def list_models(self) -> Dict[str, ModelInfo]:
        """列出所有模型"""
        return self.models_config.copy()
    
    def add_model(self, model_id: str, config: Dict[str, Any]) -> None:
        """添加新模型配置"""
        model_info = ModelInfo(
            name=config['name'],
            model_type=config['type'],
            pipeline_class=config['pipeline_class'],
            config=config,
            memory_usage=config.get('memory_usage', 'medium')
        )
        
        self.models_config[model_id] = model_info
        self.save_config()
        logger.info(f"添加新模型配置: {model_id}")
    
    def update_model(self, model_id: str, config: Dict[str, Any]) -> None:
        """更新模型配置"""
        if model_id not in self.models_config:
            raise ValueError(f"模型不存在: {model_id}")
            
        # 如果是当前加载的模型，需要先卸载
        if self.current_model == model_id:
            with self.model_lock:
                self._unload_current_model()
                
        # 更新配置
        model_info = self.models_config[model_id]
        model_info.config.update(config)
        model_info.name = config.get('name', model_info.name)
        model_info.model_type = config.get('type', model_info.model_type)
        model_info.pipeline_class = config.get('pipeline_class', model_info.pipeline_class)
        model_info.memory_usage = config.get('memory_usage', model_info.memory_usage)
        
        self.save_config()
        logger.info(f"更新模型配置: {model_id}")
    
    def remove_model(self, model_id: str) -> None:
        """删除模型配置"""
        if model_id not in self.models_config:
            raise ValueError(f"模型不存在: {model_id}")
            
        # 如果是当前加载的模型，需要先卸载
        if self.current_model == model_id:
            with self.model_lock:
                self._unload_current_model()
                
        del self.models_config[model_id]
        self.save_config()
        logger.info(f"删除模型配置: {model_id}")
    
    def get_current_model_status(self) -> Dict[str, Any]:
        """获取当前模型状态"""
        with self.model_lock:
            if self.current_model is None:
                return {
                    'current_model': None,
                    'is_loaded': False,
                    'last_used': None,
                    'gpu_memory': self._get_gpu_memory_info()
                }
                
            model_info = self.models_config.get(self.current_model)
            return {
                'current_model': self.current_model,
                'model_name': model_info.name if model_info else 'Unknown',
                'is_loaded': model_info.is_loaded if model_info else False,
                'last_used': model_info.last_used.isoformat() if model_info and model_info.last_used else None,
                'gpu_memory': self._get_gpu_memory_info()
            }
    
    def _get_gpu_memory_info(self) -> Dict[str, float]:
        """获取GPU内存信息"""
        if not torch.cuda.is_available():
            return {'allocated': 0.0, 'cached': 0.0, 'total': 0.0}
            
        return {
            'allocated': torch.cuda.memory_allocated(0) / 1024**3,
            'cached': torch.cuda.memory_reserved(0) / 1024**3,
            'total': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }