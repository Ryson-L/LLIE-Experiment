import os
import cv2
import numpy as np
from glob import glob
from skimage.metrics import structural_similarity as ssim
import re

# ================================
# CONFIG
# ================================
RESULT_DIR = '../experiment/test2/results'
GT_DIR     = '../dataset/test/high'
MASK_DIR   = '../dataset/test/masks_person'

E_START = 2   # 起始 epoch
E_END   = 17   # 结束 epoch
# ================================

def extract_epoch_and_id(fname):
    """
    输入: Epoch3_0186_x4_HR_Pred.png
    输出: epoch=3, id=0186
    """
    m = re.match(r"Epoch(\d+)_([0-9]+)_", fname)
    if m is None:
        return None, None
    return int(m.group(1)), m.group(2)


def calculate_psnr(img1, img2, mask=None):
    diff = img1 - img2
    if mask is not None:
        diff = diff * mask
        valid = np.sum(mask) * 3
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


def evaluate():
    # 找到所有结果图
    files = sorted(glob(os.path.join(RESULT_DIR, "*.png")))

    # 按 epoch 归类
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

    # 最终 epoch 平均
    final_gpsnr, final_mpsnr = [], []
    final_gssim, final_mssim = [], []

    for ep in sorted(epoch_buckets.keys()):
        files = epoch_buckets[ep]

        g_psnr_list = []
        m_psnr_list = []
        g_ssim_list = []
        m_ssim_list = []

        for img_id, f_enh in files:
            # GT / Mask
            gt_path   = os.path.join(GT_DIR,   img_id + ".png")
            mask_path = os.path.join(MASK_DIR, img_id + ".png")

            # Read images
            enh = cv2.imread(f_enh).astype(np.float32) / 255.0
            gt  = cv2.imread(gt_path).astype(np.float32) / 255.0

            if enh.shape != gt.shape:
                enh = cv2.resize(enh, (gt.shape[1], gt.shape[0]))

            # Mask
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.float32)
                if mask.shape != gt.shape[:2]:
                    mask = cv2.resize(mask, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask = np.expand_dims(mask, axis=2)
            else:
                mask = None

            g_psnr = calculate_psnr(enh, gt, mask=None)
            g_ssim = calculate_ssim(enh, gt, mask=None)

            if mask is not None:
                m_psnr = calculate_psnr(enh, gt, mask)
                m_ssim = calculate_ssim(enh, gt, mask)
            else:
                m_psnr = 0
                m_ssim = 0

            g_psnr_list.append(g_psnr)
            m_psnr_list.append(m_psnr)
            g_ssim_list.append(g_ssim)
            m_ssim_list.append(m_ssim)

        # 统计当前 epoch
        ep_gpsnr = np.mean(g_psnr_list)
        ep_mpsnr = np.mean(m_psnr_list)
        ep_gssim = np.mean(g_ssim_list)
        ep_mssim = np.mean(m_ssim_list)

        print(f"Epoch {ep:3d}:  G-PSNR={ep_gpsnr:.4f}  M-PSNR={ep_mpsnr:.4f}  |  G-SSIM={ep_gssim:.4f}  M-SSIM={ep_mssim:.4f}")

        final_gpsnr.append(ep_gpsnr)
        final_mpsnr.append(ep_mpsnr)
        final_gssim.append(ep_gssim)
        final_mssim.append(ep_mssim)

    # 最终平均（跨 epoch）
    print("----------------------------------------------------------")
    print(f"AVERAGED ({E_START}~{E_END}):")
    print(f"Global   PSNR: {np.mean(final_gpsnr):.4f}")
    print(f"Portrait PSNR: {np.mean(final_mpsnr):.4f}")
    print(f"Global   SSIM: {np.mean(final_gssim):.4f}")
    print(f"Portrait SSIM: {np.mean(final_mssim):.4f}")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    evaluate()
