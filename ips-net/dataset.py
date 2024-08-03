import torch
from torchvision import transforms
from PIL import Image
from torch.utils import data
import os
import random
import torchvision.transforms.functional as TF
import torchvision

def augment(img, gt):
    augmentation_method = random.choice([0, 1, 2, 3, 4, 5, 6, 7])

    '''Rotate'''
    if augmentation_method == 0 or augmentation_method == 1:
        t = transforms.RandomRotation(degrees=[-90, 90])
        rotate_degree = t.get_params(degrees=[-90, 90])
        img = TF.rotate(img, rotate_degree)
        gt = TF.rotate(gt, rotate_degree)
        return img, gt

    '''Vertical'''
    if augmentation_method == 2:
        vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
        img = vertical_flip(img)
        gt = vertical_flip(gt)
        return img, gt

    '''Horizontal'''
    if augmentation_method == 3:
        horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
        img = horizontal_flip(img)
        gt = horizontal_flip(gt)
        return img, gt

    '''Crop'''
    if augmentation_method == 4:
        random_crop = transforms.RandomCrop((64, 64))
        i, j, h, w = random_crop.get_params(img, (64, 64))
        res = transforms.Resize((256, 256))
        img = res(TF.crop(res(img), i, j, h, w))
        gt = res(TF.crop(res(gt), i, j, h, w))
        return img, gt

    '''Noise'''
    if augmentation_method == 5:
        for i in range(round(img.size[0] * img.size[1] / 5)):
            img.putpixel(
                (random.randint(0, img.size[0] - 1), random.randint(0, img.size[1] - 1)),
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            )
        return img, gt

    '''no change'''
    if augmentation_method == 6 or augmentation_method == 7:
        return img, gt




class SolarDataset(data.Dataset):
    def __init__(self, data_names, train_mode='Train', augmentation=False):
        original = []
        ref = []
        illum = []
        masks = []   
        clahe = []     
        for name in data_names:
            files = os.listdir(f'./data/{train_mode}/{name}/images')
            for file in files:
                original.append(f'./data/{train_mode}/{name}/images/{file}')
                ref.append(f'./data/{train_mode}/{name}/decmp/R1/{file}')
                illum.append(f'./data/{train_mode}/{name}/decmp/L1/{file}')
                clahe.append(f'./data_CLAHE/{train_mode}/{name}/images/{file}')
                masks.append(f'./data/{train_mode}/{name}/masks/{file}')
        
        self.original = sorted(original)
        self.ref = sorted(ref)
        self.illum = sorted(illum)
        self.masks = sorted(masks)
        self.clahe = sorted(clahe)
        self.augmentation = augmentation
        self.transform = transforms.Compose([transforms.Resize(256), 
                                                     transforms.PILToTensor(),
                                                     transforms.ConvertImageDtype(torch.float)])

    def __getitem__(self, index):
        
        img_path = self.original[index]
        illum_path = self.illum[index]
        ref_path = self.ref[index]
        cl_path = self.clahe[index]
        gt_path = self.masks[index]
        
        name = self.original[index].split('/')[-1].split('.')[0]
       
        img = Image.open(img_path).convert('RGB')
        illum = Image.open(illum_path).convert('RGB')
        ref = Image.open(ref_path).convert('RGB')
        cl = Image.open(cl_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')
        
        h = img.size[0]
        w = img.size[1]
        
        img = self.transform(img)
        illum = self.transform(illum)
        ref = self.transform(ref)
        cl = self.transform(cl)
        gt = self.transform(gt)
        
        if self.augmentation:
            image, gt = augment(img, gt)
                
        sample = {"index": index, "image": img, "illum": illum, "ref" : ref, "clahe": cl, "label": gt[0, :, :].unsqueeze(0), "original": img, 'img_path': img_path, 'illum_path' : illum_path,'ref_path' : ref_path,'gt_path': gt_path, "height": h, "width": w, 'name': name}

        return sample

    def __len__(self):
        return len(self.original)
        
