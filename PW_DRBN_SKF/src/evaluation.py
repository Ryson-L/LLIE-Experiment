import os
import cv2
import numpy as np
from glob import glob
from skimage.metrics import structural_similarity as ssim

# ================= 配置区域 =================
# 1. 待评估的增强图路径 (Input)
ENHANCED_DIR = '../experiment/test2/epoch18' 

# 2. 地面真值 (GT) 路径 (Reference)
GT_DIR = '../dataset/test/high'

# 3. 二值掩码路径 (Weight)
MASK_DIR = '../dataset/test/masks_person'
# ===========================================

def calculate_psnr(img1, img2, mask=None):
    diff = img1 - img2
    if mask is not None:
        diff = diff * mask
        valid_pixels = np.sum(mask) * 3
        if valid_pixels == 0: return 0.0 
        mse = np.sum(diff ** 2) / valid_pixels
    else:
        mse = np.mean(diff ** 2)

    if mse == 0: return 100.0
    return -10.0 * np.log10(mse)

def calculate_ssim(img1, img2, mask=None):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 确保尺寸一致（防止因为裁剪导致尺寸不匹配报错）
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    if mask is None:
        return ssim(gray1, gray2, data_range=1.0)
    
    mask_gray = mask[:, :, 0]
    # 确保 mask 尺寸也一致
    if mask_gray.shape != gray1.shape:
        mask_gray = cv2.resize(mask_gray, (gray1.shape[1], gray1.shape[0]), interpolation=cv2.INTER_NEAREST)

    _, ssim_map = ssim(gray1, gray2, full=True, data_range=1.0)
    
    valid_pixels = np.sum(mask_gray)
    if valid_pixels == 0: return 0.0
    
    masked_ssim = np.sum(ssim_map * mask_gray) / valid_pixels
    return masked_ssim

def evaluate():
    if not os.path.exists(ENHANCED_DIR):
        print(f"Error: Enhanced directory not found: {ENHANCED_DIR}")
        return

    # 获取所有增强图
    files = sorted(glob(os.path.join(ENHANCED_DIR, '*.*')))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not files:
        print("No images found in ENHANCED_DIR.")
        return

    print(f"Evaluating {len(files)} images from: {ENHANCED_DIR}")
    print(f"{'Image ID':<15} | {'G-PSNR':<8} {'M-PSNR':<8} | {'G-SSIM':<8} {'M-SSIM':<8}")
    print("-" * 65)

    metrics = {'g_psnr': [], 'm_psnr': [], 'g_ssim': [], 'm_ssim': []}

    for fpath in files:
        filename = os.path.basename(fpath)
        
        # --- 核心修改：文件名解析 ---
        # 格式: Epoch143_0186_x4_HR_Pred.png
        # 我们需要提取中间的 '0186'
        try:
            # 以 '_' 分割，取索引 1 的部分
            parts = filename.split('_')
            # 简单校验：找到像是ID的部分
            real_id = parts[1] 
            
            # 构造 GT 文件名 (假设 GT 是 .png，如果不是请修改后缀)
            gt_filename = real_id + '.png'
            mask_filename = real_id + '.png'
            
        except Exception as e:
            print(f"Skipping {filename}: Parsing failed ({e})")
            continue

        gt_path = os.path.join(GT_DIR, gt_filename)
        mask_path = os.path.join(MASK_DIR, mask_filename)

        if not os.path.exists(gt_path):
            # 尝试 jpg 后缀以防万一
            gt_path = os.path.join(GT_DIR, real_id + '.jpg')
            if not os.path.exists(gt_path):
                print(f"Skipping {gt_filename}: GT not found (ID: {real_id})")
                continue
        
        # 1. 读取 & 归一化
        img_enh = cv2.imread(fpath).astype(np.float32) / 255.0
        img_gt  = cv2.imread(gt_path).astype(np.float32) / 255.0
        
        # 尺寸校验 (有些增强可能会改变尺寸)
        if img_enh.shape != img_gt.shape:
            img_enh = cv2.resize(img_enh, (img_gt.shape[1], img_gt.shape[0]))

        # 2. 读取 Mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            # 调整 Mask 尺寸以匹配图片
            if mask.shape != img_gt.shape[:2]:
                 mask = cv2.resize(mask, (img_gt.shape[1], img_gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=2)
        else:
            mask = None

        # 3. 计算
        g_psnr = calculate_psnr(img_enh, img_gt, mask=None)
        g_ssim = calculate_ssim(img_enh, img_gt, mask=None)
        
        if mask is not None:
            m_psnr = calculate_psnr(img_enh, img_gt, mask)
            m_ssim = calculate_ssim(img_enh, img_gt, mask)
        else:
            m_psnr, m_ssim = 0.0, 0.0

        # 4. 记录
        metrics['g_psnr'].append(g_psnr)
        metrics['g_ssim'].append(g_ssim)
        if mask is not None:
            metrics['m_psnr'].append(m_psnr)
            metrics['m_ssim'].append(m_ssim)

        print(f"{real_id:<15} | {g_psnr:.2f}     {m_psnr:.2f}     | {g_ssim:.4f}   {m_ssim:.4f}")

    print("-" * 65)
    print("SCORES:")
    print(f"Global   PSNR: {np.mean(metrics['g_psnr']):.4f}")
    print(f"Portrait PSNR: {np.mean(metrics['m_psnr']):.4f}")
    print(f"Global   SSIM: {np.mean(metrics['g_ssim']):.4f}")
    print(f"Portrait SSIM: {np.mean(metrics['m_ssim']):.4f}")
    print("-" * 65)

if __name__ == "__main__":
    evaluate()