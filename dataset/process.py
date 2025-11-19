import os
from PIL import Image

# ================= 配置区域 =================
# 输入文件夹：放那 90 张原图
INPUT_FOLDER = "raw_downloads"

# 输出文件夹：处理好的 512x512 高光图 (Ground Truth)
OUTPUT_FOLDER = "dataset_512/high" 

# 目标尺寸
TARGET_SIZE = 512
# ===========================================

def process_images():
    # 如果输出目录不存在，自动创建
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 支持的图片格式
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    
    files = [f for f in os.listdir(INPUT_FOLDER) if os.path.splitext(f)[1].lower() in valid_exts]
    # 排序一下，保证每次运行顺序一致
    files.sort()

    print(f"找到 {len(files)} 张图片，准备处理...")

    success_count = 0
    for index, filename in enumerate(files):
        img_path = os.path.join(INPUT_FOLDER, filename)
        
        try:
            with Image.open(img_path) as img:
                # 1. 转为 RGB (防止某些图片是 CMYK 或 RGBA 导致报错)
                img = img.convert('RGB')
                width, height = img.size

                # 2. 计算中心裁剪区域 (切出最大的正方形)
                min_dim = min(width, height)
                left = (width - min_dim) // 2
                top = (height - min_dim) // 2
                right = (width + min_dim) // 2
                bottom = (height + min_dim) // 2
                
                # 裁剪
                img_cropped = img.crop((left, top, right, bottom))
                
                # 3. 高质量缩放到 512x512 (使用 Lanczos 算法抗锯齿)
                img_resized = img_cropped.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
                
                # 4. 保存为 PNG
                # 格式化文件名: 0001.png, 0002.png ...
                new_filename = f"{success_count + 1:04d}.png"
                save_path = os.path.join(OUTPUT_FOLDER, new_filename)
                
                img_resized.save(save_path, format='PNG')
                
                print(f"[{index+1}/{len(files)}] {filename} -> {new_filename} (Size: {img_resized.size})")
                success_count += 1

        except Exception as e:
            print(f"!!! 错误: 处理 {filename} 失败. 原因: {e}")

    print(f"\n处理完成！")
    print(f"成功转换: {success_count} 张")
    print(f"文件保存在: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    process_images()