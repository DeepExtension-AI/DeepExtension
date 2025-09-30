from diffusers import DiffusionPipeline
import torch

# 设置设备
device = "cuda:0"
# print(f"使用设备: {device}")

# 加载基础模型
print("Loading base model...")
pipe = DiffusionPipeline.from_pretrained(
    "/home/cicd/workspace/fine-tuning-image/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16,
    device_map="balanced"
)

# 加载LoRA权重
print("Loading LoRA weights...")
pipe.load_lora_weights("/home/cicd/workspace/fine-tuning-image/SimpleTuner/output/models/2025080401", weight_name="pytorch_lora_weights.safetensors")

# 融合LoRA参数
print("Fusing LoRA weights...")
pipe.fuse_lora(lora_scale=0.7)

# 推理
print("Running inference...")
image = pipe(
    prompt="A futuristic city skyline at sunset, ultra-detailed",
    negative_prompt="blurry, low quality",
    num_inference_steps=28,  # SD3推荐28-35步
    guidance_scale=5.0,      # SD3推荐5-7范围
).images[0]

# 保存图像
image.save("output_123.png")