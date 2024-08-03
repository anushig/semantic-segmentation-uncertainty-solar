import os
from PIL import Image

imgs_folder = "./images/"
masks_folder = "./masks/"
illum_folder = "./decmp/L1/"
ref_folder = "./decmp/R1/"

img_files = os.listdir(imgs_folder)

for img_file in img_files:
    img_path = os.path.join(imgs_folder, img_file)
    mask_path = os.path.join(masks_folder, img_file)
    illum_path = os.path.join(illum_folder, img_file)
    ref_path = os.path.join(ref_folder, img_file)
    
    try:
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(mask_path) as mask:
            mask.verify()
        with Image.open(illum_path) as illum:
            illum.verify()   
        with Image.open(ref_path) as ref:
            ref.verify()
            
    
    except (IOError, SyntaxError) as e:
        os.remove(img_path)
        os.remove(mask_path)
        os.remove(illum_path)
        os.remove(ref_path)
        print(f"Corrupted image and mask deleted: {img_file}")

print("Corrupted images and masks have been deleted.")

