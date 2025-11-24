import os
import torch
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# import HRSeg model, reuse its functions
from hrseg.hrseg_model import create_hrnet



TEST_GT_DIR = '../dataset/test/high'

SAVE_DIR = '../dataset/test/masks_person'

TARGET_CLASS_INDEX = 36 #id for person class in Pascal VOC
# ===========================================

def generate_masks():

    if not os.path.exists(TEST_GT_DIR):
        print(f"Error: Input directory {TEST_GT_DIR} not found.")
        return
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created output directory: {SAVE_DIR}")

    img_paths = sorted(glob(os.path.join(TEST_GT_DIR, '*.*')))
    print(f"Found {len(img_paths)} images. Starting mask generation...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = create_hrnet().cuda()
    model.eval()

    print(f"Model loaded. Extracting class index {TARGET_CLASS_INDEX}...")

    with torch.no_grad():
        for img_path in tqdm(img_paths):
            filename = os.path.basename(img_path)

            img_pil = Image.open(img_path).convert('RGB')
            w, h = img_pil.size
            input_tensor = transform(img_pil).unsqueeze(0).cuda()

            out = model(input_tensor)
            if isinstance(out, (list, tuple)): out = out[0]

            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            
            pred_indices = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # generate binary mask for the person class
            person_mask = np.zeros_like(pred_indices, dtype=np.uint8)
            person_mask[pred_indices == TARGET_CLASS_INDEX] = 255
            
            save_name = os.path.splitext(filename)[0] + '.png'
            cv2.imwrite(os.path.join(SAVE_DIR, save_name), person_mask)

    print("-" * 50)
    print("Generation Complete.")
    print(f"Binary masks saved to: {SAVE_DIR}")
    print("Ready for evaluation.")

if __name__ == "__main__":
    generate_masks()