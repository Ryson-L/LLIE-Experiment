import os

val_dir = './experiments/Portrait_SKF/val_images'

for fname in os.listdir(val_dir):
    # 只管那些长得像 EpochXXX_XXX.png_Pred.png 的
    if '.png_Pred.png' not in fname:
        continue

    old_path = os.path.join(val_dir, fname)

    # 拆掉最后一个 .png 得到 "Epoch001_0186.png_Pred"
    base, ext = os.path.splitext(fname)  # ext == '.png'
    # 把中间那段 ".png_Pred" 变成 "_Pred"
    new_base = base.replace('.png_Pred', '_Pred')  # "Epoch001_0186_Pred"
    new_fname = new_base + ext                     # "Epoch001_0186_Pred.png"

    new_path = os.path.join(val_dir, new_fname)
    print(f'{fname}  ->  {new_fname}')
    os.rename(old_path, new_path)
