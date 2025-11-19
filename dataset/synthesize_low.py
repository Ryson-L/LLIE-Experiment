import cv2
import numpy as np
import os
import random

# ================= 配置区域 =================
# 输入：上一处理好的 512x512 高光图
INPUT_FOLDER = os.path.join("dataset_512", "high")

# 输出：生成的低光图
OUTPUT_FOLDER = os.path.join("dataset_512", "low")

# 物理参数设置 (可以微调，但现在的默认值比较符合真实低光)
# 1. 变暗范围 (模拟曝光不足): 0.1 到 0.3 (越小越黑)
DARKNESS_RANGE = (0.5, 0.7)

# 2. Gamma 范围 (模拟非线性响应): 2.0 到 3.0
GAMMA_RANGE = (1.5, 2.2)

# 3. 散粒噪声参数 (Poisson Noise - 与光量有关): 模拟光子随机性
SHOT_NOISE_SCALE = (0.0005, 0.0015)

# 4. 读出噪声参数 (Read Noise - 高斯分布): 模拟电路底噪
READ_NOISE_SCALE = (0.005, 0.010)
# ===========================================

def synthesize_low_light():
    if not os.path.exists(INPUT_FOLDER):
        print(f"错误：找不到输入文件夹 {INPUT_FOLDER}")
        return
        
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 获取所有图片
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort() # 保证顺序一致

    print(f"开始处理 {len(files)} 张图片...")

    for idx, filename in enumerate(files):
        # 1. 读取图片 (OpenCV 读取的是 BGR 格式，0-255)
        img_path = os.path.join(INPUT_FOLDER, filename)
        img_high = cv2.imread(img_path)

        if img_high is None:
            continue

        # 转为 float32 并归一化到 [0, 1] 进行数学运算
        img_high = img_high.astype(np.float32) / 255.0

        # ================= 核心物理退化过程 =================
        
        # Step A: 模拟光线不足 (Linear Dimming & Gamma)
        # 随机生成变暗系数和 Gamma 值
        dark_factor = random.uniform(*DARKNESS_RANGE)
        gamma = random.uniform(*GAMMA_RANGE)
        
        # 公式: I_low = (I_high * dark_factor) ^ gamma
        img_low = np.power(img_high * dark_factor, gamma)

        # Step B: 注入物理噪声 (Physics-based Noise Injection)
        # 根据 Wei et al. (CVPR 2020) 的定义：Noise = Poisson(Signal) + Gaussian
        
        # B1. 散粒噪声 (Photon Shot Noise) - 依赖于信号强度
        # 信号越强(越亮)，噪声方差越大。这里用高斯近似泊松分布 N(0, sigma_s * sqrt(I))
        shot_scale = random.uniform(*SHOT_NOISE_SCALE)
        # np.maximum 确保根号下非负
        shot_noise = np.random.normal(scale=shot_scale * np.sqrt(np.maximum(img_low, 1e-10)))
        
        # B2. 读出噪声 (Read Noise) - 独立的高斯噪声
        read_scale = random.uniform(*READ_NOISE_SCALE)
        read_noise = np.random.normal(scale=read_scale, size=img_low.shape)
        
        # 叠加噪声
        img_low_noisy = img_low + shot_noise + read_noise

        # ===================================================

        # 3. 截断到有效范围 [0, 1] 并转回 8位整数
        img_low_noisy = np.clip(img_low_noisy, 0.0, 1.0)
        img_final = (img_low_noisy * 255).astype(np.uint8)

        # 4. 保存 (文件名保持一致，例如 0001.png)
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, img_final)

        if (idx + 1) % 10 == 0:
            print(f"已处理: {idx + 1}/{len(files)}")

    print(f"\n全部完成！低光数据集保存在: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    synthesize_low_light()