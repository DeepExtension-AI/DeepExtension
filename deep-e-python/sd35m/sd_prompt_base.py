import torch
from diffusers import DiffusionPipeline
import argparse

# 预定义的常用比例
ASPECT_RATIOS = {
    "1:1": (1024, 1024),      # 正方形
    "16:9": (1152, 648),      # 宽屏
    "9:16": (648, 1152),      # 竖屏
    "4:3": (1024, 768),       # 传统
    "3:4": (768, 1024),       # 传统竖屏
    "2:3": (832, 1248),       # 肖像
    "3:2": (1248, 832),       # 风景
    "21:9": (1344, 576),      # 超宽屏
}

def calculate_dimensions(aspect_ratio, base_size=1024):
    """根据比例计算宽高"""
    if ":" in aspect_ratio:
        width_ratio, height_ratio = map(int, aspect_ratio.split(":"))
        # 保持较大的边为base_size
        if width_ratio >= height_ratio:
            width = base_size
            height = int(base_size * height_ratio / width_ratio)
        else:
            height = base_size
            width = int(base_size * width_ratio / height_ratio)
        
        # 确保尺寸是8的倍数（模型要求）
        width = (width // 8) * 8
        height = (height // 8) * 8
        return width, height
    else:
        return ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))

def main():
    # 设置参数解析
    parser = argparse.ArgumentParser(description='Stable Diffusion 3.5 Inference')
    parser.add_argument('--model_path', type=str, help='Path to base model', required=True)
    parser.add_argument('--prompt', type=str, help='Positive prompt', required=True)
    parser.add_argument('--negative_prompt', type=str, default="blurry, low quality", help='Negative prompt')
    parser.add_argument('--output', type=str, help='Output image filename', required=True)
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--aspect_ratio', type=str, default="1:1", help='Aspect ratio (e.g., 16:9, 4:3, 9:16) or preset name')
    parser.add_argument('--base_size', type=int, default=1024, help='Base size for calculating dimensions')
    parser.add_argument('--num_inference_steps', type=int, default=28, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=5.0, help='Guidance scale')
    
    known_args, unknown_args = parser.parse_known_args()
    print("Known args:", known_args)
    print("Unknown args:", unknown_args)
    args = known_args

    # 计算宽高尺寸
    width, height = calculate_dimensions(args.aspect_ratio, args.base_size)
    print(f"Aspect ratio: {args.aspect_ratio}, Dimensions: {width}x{height}")

    # 加载基础模型
    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="balanced"
    )

    # 生成多张图片
    print(f"Generating {args.batch_size} images...")
    for i in range(args.batch_size):
        # 推理
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            width=width,
            height=height
        ).images[0]

        # 保存图像（如果生成多张，添加序号）
        if args.batch_size > 1:
            name, ext = args.output.rsplit('.', 1)
            output_filename = f"{name}_{i+1:03d}.{ext}"
        else:
            output_filename = args.output
            
        image.save(output_filename)
        print(f"Saved: {output_filename} ({width}x{height})")

    print("Inference completed!")

if __name__ == "__main__":
    main()