# 动态模型管理图像生成服务

基于原有 `flux_release_edit_0806.py` 重新设计的动态模型管理系统，支持多种AI图像生成模型的动态加载、自动卸载和配置管理。

## 🚀 主要特性

### 1. 动态模型管理
- **冷加载机制**: 按需加载模型到GPU显存
- **自动卸载**: 30分钟无使用自动释放显存
- **智能切换**: 确保显存中同时只保留一个模型
- **时间戳记录**: 跟踪每个模型的最后使用时间

### 2. 多模型支持
- **FLUX系列**: FLUX.1-schnell, FLUX.1-Kontext-dev等
- **Stable Diffusion**: 包括SD 3.5 Medium
- **GGUF格式**: 支持bullerwins/FLUX.1-Kontext-dev-GGUF
- **Nunchaku优化**: 支持本地和HuggingFace优化模型
- **扩展性**: 通用接口设计，便于添加新模型

### 3. 配置管理
- **YAML配置**: 统一的模型配置文件
- **RESTful API**: 完整的CRUD操作接口
- **热更新**: 运行时修改配置无需重启
- **配置验证**: 自动验证配置文件格式

### 4. 高级功能
- **任务队列**: 异步处理生成请求
- **优先级支持**: 任务优先级管理
- **文件管理**: 自动清理过期文件
- **健康检查**: 系统状态监控
- **线程安全**: 完整的并发控制

## 📋 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA 11.8+
- **显存**: 至少8GB (推荐16GB+)
- **内存**: 至少16GB RAM
- **存储**: 至少50GB可用空间

## 🛠️ 安装配置

### 1. 环境准备

```bash
# 克隆或复制项目文件
cd /path/to/your/project

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置文件

编辑 `models_config.yaml` 文件，配置你的模型:

```yaml
models:
  flux.1-schnell:
    name: "FLUX.1 Schnell"
    type: "flux"
    pipeline_class: "FluxPipeline"
    huggingface_path: "black-forest-labs/FLUX.1-schnell"
    local_path: "flux_models/FLUX.1-schnell"
    # ... 其他配置
```

### 3. 环境变量

设置必要的环境变量:

```bash
export HF_TOKEN="your_huggingface_token"
export CUDA_VISIBLE_DEVICES="0"  # 指定GPU
```

### 4. 启动服务

```bash
# 开发模式
python dynamic_flux_app.py

# 生产模式 (推荐)
pip install waitress
python dynamic_flux_app.py
```

服务将在 `http://0.0.0.0:5050` 启动。

## 📚 API 文档

### 图像生成接口

#### 文生图 (Text-to-Image)
```http
POST /text2image
Content-Type: application/json

{
  "prompt": "A beautiful landscape",
  "model_name": "flux_dev",
  "batch_size": 1,
  "aspect_ratio": "16:9",
  "priority": 0
}
```

#### 图生图 (Image-to-Image)
```http
POST /image2image
Content-Type: application/json

{
  "prompt": "Transform this image",
  "model_name": "flux_dev",
  "base64_image": "data:image/png;base64,...",
  "strength": 0.8,
  "batch_size": 1,
  "aspect_ratio": "1:1"
}
```

### 任务管理接口

#### 查询任务状态
```http
GET /task/{task_id}
```

#### 列出所有任务
```http
GET /tasks?status=completed&limit=50
```

#### 清空任务队列
```http
POST /queue/clear
```

### 模型管理接口

#### 列出所有模型
```http
GET /models
```

#### 获取模型信息
```http
GET /models/{model_id}
```

#### 更新模型配置
```http
PUT /models/{model_id}
Content-Type: application/json

{
  "name": "Updated Model Name",
  "memory_usage": "high",
  "description": "Updated description"
}
```

#### 删除模型配置
```http
DELETE /models/{model_id}
```

#### 添加新模型
```http
POST /models
Content-Type: application/json

{
  "model_id": "new_model",
  "name": "New Model",
  "type": "flux",
  "pipeline_class": "FluxPipeline",
  "hf_path": "path/to/model"
}
```

#### 获取当前模型状态
```http
GET /models/current
```

#### 手动卸载当前模型
```http
POST /models/unload
```

### 系统监控接口

#### 健康检查
```http
GET /health
```

返回系统状态、GPU信息、队列状态等。

## 🏗️ 系统架构

### 核心组件

1. **DynamicModelManager**: 模型动态管理核心
   - 模型加载/卸载
   - 内存管理
   - 配置管理
   - 线程安全

2. **ModelLoader**: 抽象模型加载器
   - FluxModelLoader
   - NunchakuModelLoader  
   - StableDiffusionModelLoader
   - GGUFModelLoader

3. **TaskManager**: 任务队列管理
   - 优先级队列
   - 异步处理
   - 状态跟踪

4. **Flask API**: RESTful接口层
   - 图像生成接口
   - 模型管理接口
   - 系统监控接口

### 设计模式

- **工厂模式**: ModelLoader的创建
- **单例模式**: DynamicModelManager实例
- **策略模式**: 不同模型的加载策略
- **观察者模式**: 模型状态变化通知
- **命令模式**: 任务队列处理

### 线程安全

- **模型锁**: 确保模型操作的原子性
- **队列锁**: 保护任务队列的并发访问
- **配置锁**: 配置文件读写的线程安全
- **内存锁**: GPU内存操作的同步

## 🔧 配置说明

### models_config.yaml 结构

```yaml
# 全局设置
global:
  auto_unload_minutes: 30
  default_model: "flux_dev"
  gpu_memory_fraction: 0.9
  enable_attention_slicing: true
  enable_vae_slicing: true
  enable_cpu_offload: true
  log_level: "INFO"

# 模型配置
models:
  model_id:
    name: "显示名称"
    type: "模型类型 (flux/stable_diffusion/gguf)"
    pipeline_class: "管道类名"
    hf_path: "HuggingFace路径"
    local_path: "本地路径 (可选)"
    torch_dtype: "数据类型 (float16/bfloat16)"
    memory_usage: "内存使用 (low/medium/high)"
    supports_text2img: true
    supports_img2img: false
    description: "模型描述"
```

### 支持的宽高比

- `1:1` - 正方形 (768x768)
- `16:9` - 宽屏 (1365x768)
- `9:16` - 竖屏 (768x1365)
- `4:3` - 标准 (1024x768)
- `3:4` - 竖版标准 (768x1024)
- `21:9` - 超宽屏 (1792x768)
- `9:21` - 超竖屏 (768x1792)

## 🚨 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 启用CPU offload
   - 使用更小的模型

2. **模型加载失败**
   - 检查HuggingFace token
   - 验证模型路径
   - 查看日志详细错误

3. **任务处理缓慢**
   - 检查GPU利用率
   - 优化模型配置
   - 调整队列优先级

4. **配置文件错误**
   - 验证YAML语法
   - 检查必需字段
   - 查看启动日志

### 日志查看

```bash
# 查看应用日志
tail -f /path/to/logfile

# 查看GPU状态
nvidia-smi

# 查看系统资源
htop
```

## 🔒 安全注意事项

1. **API访问控制**: 建议在生产环境中添加认证
2. **文件权限**: 确保输出目录权限正确
3. **资源限制**: 设置合理的内存和GPU使用限制
4. **网络安全**: 使用防火墙限制访问端口
5. **Token安全**: 妥善保管HuggingFace token

## 📈 性能优化

### GPU优化
- 启用attention slicing
- 启用VAE slicing
- 使用CPU offload
- 合理设置torch_dtype

### 内存优化
- 及时清理未使用模型
- 使用内存映射
- 优化批处理大小
- 启用梯度检查点

### 网络优化
- 使用CDN缓存模型
- 启用模型压缩
- 优化数据传输
- 使用异步处理

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Black Forest Labs FLUX](https://github.com/black-forest-labs/flux)
- [Stability AI](https://stability.ai/)
- 原始 `flux_release_edit_0806.py` 的开发者

## 📞 支持

如有问题或建议，请:
1. 查看文档和FAQ
2. 搜索已有的Issues
3. 创建新的Issue
4. 联系维护者

---

**注意**: 本系统需要大量GPU资源，请确保硬件配置满足要求。建议在测试环境中先进行充分验证后再部署到生产环境。