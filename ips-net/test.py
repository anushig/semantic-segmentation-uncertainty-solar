import torch
import numpy as np
from torchvision.utils import save_image
from metrics import calculate_metrics
import json
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
m = torch.nn.Sigmoid()
bce_loss = torch.nn.BCEWithLogitsLoss()

def validate(loader, net, exp_name, e, val_name):
    net.eval()
    iou_score = []
    losses = []
    f1_score = []

    val_iou_scores = {}    
    for d in loader:
        img = d['image'].to(device)
        illum = d["illum"].to(device)
        ref = d["ref"].to(device)
        cl = d["clahe"].to(device)
        gt = d['label'].to(device)
        
        with torch.no_grad():
            output = net.forward(img,ref,illum, cl,"Test")

        loss = bce_loss(output[0], gt) 
        losses.append(loss.item())
        output_ = m(output[0])
        output_ = torch.where(output_> 0.5, 1, 0)
 
        iou, f1 = calculate_metrics(output_.cpu().numpy(), gt.cpu().numpy())

        if not np.isnan(iou):
            iou_score.append(iou)
            f1_score.append(f1)

        val_iou_scores = { "img_path" : d["img_path"], "gt_path": d["gt_path"],  "name" : d["name"], "iou" : iou, "f1_score" : f1}
        if not os.path.exists(f'./Results/{exp_name}/Predictions/{val_name}/{e}'):
            os.makedirs(f'./Results/{exp_name}/Predictions/{val_name}/{e}')
       
        if not os.path.exists(f'./Results/{exp_name}/Scores/{val_name}/{e}'):
            os.makedirs(f'./Results/{exp_name}/Scores/{val_name}/{e}')

        label_mask = torch.zeros((1,3,d['image'].shape[2], d['image'].shape[3]))
        label_mask[:,0,:,:] = output[0]
            
        file_path = f'./Results/{exp_name}/Scores/{val_name}/{e}/{d["name"][0]}.json'
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'w') as json_file:
            json.dump(val_iou_scores, json_file)

        save_image(d['original']*0.3 + label_mask*0.7, f'./Results/{exp_name}/Predictions/{val_name}/{e}/{d["name"][0]}.png')


    return iou_score, losses, f1_score

