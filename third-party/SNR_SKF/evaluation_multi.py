import os
import cv2
import numpy as np
from glob import glob
from skimage.metrics import structural_similarity as ssim
import re

import torch
import lpips   # pip install lpips

# ================================
# CONFIG
# ================================
RESULT_DIR = './experiments/Portrait_SKF/val_images'
GT_DIR     = './Portrait_Dataset/test/high'
MASK_DIR   = './Portrait_Dataset/test/masks_person'

E_START = 125   # 起始 epoch
E_END   = 137   # 结束 epoch
# ================================

# 初始化 LPIPS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LPIPS_FN = lpips.LPIPS(net='alex').to(DEVICE)   # 也可以用 net='vgg'


def extract_epoch_and_id(fname):
    m = re.search(r"Epoch(\d+)_([^_]+)_Pred\.png", fname)
    if m is None:
        return None, None
    return int(m.group(1)), m.group(2)


def calculate_psnr(img1, img2, mask=None):
    diff = img1 - img2
    if mask is not None:
        diff = diff * mask
        valid = np.sum(mask) * 3  # 3 通道
        if valid == 0:
            return 0.0
        mse = np.sum(diff ** 2) / valid
    else:
        mse = np.mean(diff ** 2)

    if mse == 0:
        return 100.0
    return -10 * np.log10(mse)


def calculate_ssim(img1, img2, mask=None):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))

    if mask is None:
        return ssim(g1, g2, data_range=1.0)

    mask_gray = mask[:, :, 0]

    if mask_gray.shape != g1.shape:
        mask_gray = cv2.resize(mask_gray, (g1.shape[1], g1.shape[0]), interpolation=cv2.INTER_NEAREST)

    _, ssim_map = ssim(g1, g2, full=True, data_range=1.0)
    valid = np.sum(mask_gray)

    if valid == 0:
        return 0.0

    return np.sum(ssim_map * mask_gray) / valid


def calculate_lpips(img1, img2, mask=None):
    """
    全局 / 区域 LPIPS.
    img1, img2: HxWx3, float32, [0,1]
    mask: HxWx1, float32, {0,1} (为 None 时表示全局)
    """
    if mask is not None:
        # 扩展到 3 通道，对 RGB 统一做 mask
        m3 = np.repeat(mask, 3, axis=2)  # HxWx3
        img1 = img1 * m3
        img2 = img2 * m3

    # HWC [0,1] -> NCHW [-1,1]
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0)
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0)
    t1 = t1.to(DEVICE).float() * 2 - 1
    t2 = t2.to(DEVICE).float() * 2 - 1

    with torch.no_grad():
        d = LPIPS_FN(t1, t2)
    return float(d.squeeze().item())


def evaluate():
    files = sorted(glob(os.path.join(RESULT_DIR, "*.png")))

    # 按 epoch 分 bucket
    epoch_buckets = {}
    for f in files:
        fname = os.path.basename(f)
        epoch, img_id = extract_epoch_and_id(fname)
        if epoch is None:
            continue
        if E_START <= epoch <= E_END:
            epoch_buckets.setdefault(epoch, []).append((img_id, f))

    if not epoch_buckets:
        print("No epoch in the specified range.")
        return

    print(f"Evaluating Epochs {E_START} ~ {E_END}")
    print("----------------------------------------------------------")

    # 跨 epoch 的最终平均
    final_gpsnr, final_mpsnr, final_bpsnr = [], [], []
    final_gssim, final_mssim, final_bssim = [], [], []
    final_lpips_g, final_lpips_m, final_lpips_b = [], [], []

    for ep in sorted(epoch_buckets.keys()):
        files = epoch_buckets[ep]

        g_psnr_list, m_psnr_list, b_psnr_list = [], [], []
        g_ssim_list, m_ssim_list, b_ssim_list = [], [], []

        lpips_g_list, lpips_m_list, lpips_b_list = [], [], []

        for img_id, f_enh in files:
            gt_path   = os.path.join(GT_DIR,   img_id + ".png")
            mask_path = os.path.join(MASK_DIR, img_id + ".png")

            enh = cv2.imread(f_enh).astype(np.float32) / 255.0
            gt  = cv2.imread(gt_path).astype(np.float32) / 255.0

            if enh.shape != gt.shape:
                enh = cv2.resize(enh, (gt.shape[1], gt.shape[0]))

            # mask: portrait = 1
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.float32)
                if mask.shape != gt.shape[:2]:
                    mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = np.expand_dims(mask, axis=2)  # HxWx1
                bg_mask = 1.0 - mask
            else:
                mask = None
                bg_mask = None

            # ------ Global 指标 ------
            g_psnr = calculate_psnr(enh, gt, mask=None)
            g_ssim = calculate_ssim(enh, gt, mask=None)
            g_psnr_list.append(g_psnr)
            g_ssim_list.append(g_ssim)

            lp_g = calculate_lpips(enh, gt, mask=None)
            lpips_g_list.append(lp_g)

            # ------ Portrait 指标 ------
            if mask is not None:
                m_psnr = calculate_psnr(enh, gt, mask)
                m_ssim = calculate_ssim(enh, gt, mask)
                lp_m   = calculate_lpips(enh, gt, mask)
            else:
                m_psnr = 0.0
                m_ssim = 0.0
                lp_m   = 0.0

            m_psnr_list.append(m_psnr)
            m_ssim_list.append(m_ssim)
            lpips_m_list.append(lp_m)

            # ------ Background 指标 ------
            if bg_mask is not None:
                b_psnr = calculate_psnr(enh, gt, bg_mask)
                b_ssim = calculate_ssim(enh, gt, bg_mask)
                lp_b   = calculate_lpips(enh, gt, bg_mask)
            else:
                b_psnr = 0.0
                b_ssim = 0.0
                lp_b   = 0.0

            b_psnr_list.append(b_psnr)
            b_ssim_list.append(b_ssim)
            lpips_b_list.append(lp_b)

        # 当前 epoch 平均
        ep_gpsnr = np.mean(g_psnr_list)
        ep_mpsnr = np.mean(m_psnr_list)
        ep_bpsnr = np.mean(b_psnr_list)

        ep_gssim = np.mean(g_ssim_list)
        ep_mssim = np.mean(m_ssim_list)
        ep_bssim = np.mean(b_ssim_list)

        ep_lp_g = np.mean(lpips_g_list)
        ep_lp_m = np.mean(lpips_m_list)
        ep_lp_b = np.mean(lpips_b_list)

        print(
            f"Epoch {ep:3d}:  "
            f"G-PSNR={ep_gpsnr:.4f}  M-PSNR={ep_mpsnr:.4f}  B-PSNR={ep_bpsnr:.4f}  |  "
            f"G-SSIM={ep_gssim:.4f}  M-SSIM={ep_mssim:.4f}  B-SSIM={ep_bssim:.4f}  |  "
            f"LPIPS-G={ep_lp_g:.4f}  LPIPS-M={ep_lp_m:.4f}  LPIPS-B={ep_lp_b:.4f}"
        )

        final_gpsnr.append(ep_gpsnr)
        final_mpsnr.append(ep_mpsnr)
        final_bpsnr.append(ep_bpsnr)

        final_gssim.append(ep_gssim)
        final_mssim.append(ep_mssim)
        final_bssim.append(ep_bssim)

        final_lpips_g.append(ep_lp_g)
        final_lpips_m.append(ep_lp_m)
        final_lpips_b.append(ep_lp_b)

    # 跨 epoch 平均
    print("----------------------------------------------------------")
    print(f"AVERAGED ({E_START}~{E_END}):")
    print(f"Global   PSNR: {np.mean(final_gpsnr):.4f}")
    print(f"Portrait PSNR: {np.mean(final_mpsnr):.4f}")
    print(f"Background PSNR: {np.mean(final_bpsnr):.4f}")
    print(f"Global   SSIM: {np.mean(final_gssim):.4f}")
    print(f"Portrait SSIM: {np.mean(final_mssim):.4f}")
    print(f"Background SSIM: {np.mean(final_bssim):.4f}")
    print(f"Global   LPIPS: {np.mean(final_lpips_g):.4f}")
    print(f"Portrait LPIPS: {np.mean(final_lpips_m):.4f}")
    print(f"Background LPIPS: {np.mean(final_lpips_b):.4f}")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    evaluate()
