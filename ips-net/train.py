import torch
from dataset import SolarDataset
import torch.optim as optim
from tqdm import tqdm
from torch import nn
import numpy as np 
from torch.utils.data import DataLoader
from test import validate 
import os
from metrics import calculate_metrics
from torch.utils.tensorboard import SummaryWriter
from ips_net import IPS_Net


writer = SummaryWriter("IPS_Net_4_4")
Epochs = 250
LR = 0.0001
batchsize = 32
data_names = ["PV01", "PV03", "PV08", "BDAPPV", "Mnacac"]

train_dataset = SolarDataset(data_names, 'Train', False)
train_loader = DataLoader(train_dataset, batchsize, shuffle=True, num_workers=1, drop_last=True)

test_loaders = []
for name in data_names:
    test_d = SolarDataset([name], 'Test', False)
    test_loader = DataLoader(test_d, batchsize, shuffle=True, num_workers=1, drop_last=True)
    test_loader.name = name
    print(test_loader.name)
    test_loaders.append(test_loader)

net = IPS_Net()

optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))
bce_loss = torch.nn.BCEWithLogitsLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

exp_name = 'IPS_Net_4_4'

if not os.path.exists(f'./Results/{exp_name}'):
    os.makedirs(f'./Results/{exp_name}/ Predictions')
    os.makedirs(f'./Results/{exp_name}/Checkpoints')
   
m = nn.Sigmoid()

for e in range(0,Epochs + 1):
    print(e)
    net.train()
    net = net.to(device)

    losses = []
    iou_scores = []
    f1_scores = []

    for d in tqdm(train_loader):
        img = d['image'].to(device)
        illum = d["illum"].to(device)
        ref = d["ref"].to(device)
        cl = d["clahe"].to(device)
        gt = d['label'].to(device)
        
        optimizer.zero_grad()
        output = net.forward(img,ref,illum, cl,"Train")
        loss = bce_loss(output[0], gt) + bce_loss(output[1], gt) + bce_loss(output[2], gt) + bce_loss(output[3], gt) + bce_loss(output[4], gt)
        losses.append(loss.item()) 
        output_ = m(output[0])
        output_ = torch.where(output_ > 0.5, 1, 0)
        iou, f1 = calculate_metrics(output_.cpu().numpy(), gt.cpu().numpy())
        if not np.isnan(iou):
            iou_scores.append(iou)
            f1_scores.append(f1)

        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss/train', np.mean(losses), e)
    writer.add_scalar('IoU/train', np.mean(iou_scores), e)
    writer.add_scalar('F1_score/train', np.mean(f1_scores), e)
    print(f'Epoch - {e}, Train IOU - {np.mean(iou_scores)}, Train F1_score - {np.mean(f1_scores)}, \
    Train LOSS - {np.mean(losses)}')
    if e % 5 == 0:
        for loader, test in zip(test_loaders, data_names):
            print(test)
            val_iou_score, val_losses, val_f1_score = validate(loader, net, exp_name, e, loader.name)
            print(
                f'Epoch - {e}, Test {loader.name} IOU - {np.mean(val_iou_score)}, '
                f'Val F1_score - {np.mean(val_f1_score)}, Val LOSS - {np.mean(val_losses)}')
            writer.add_scalar(f'IoU/val_{loader.name}', np.mean(val_iou_score), e)
            writer.add_scalar(f'Loss/val_{loader.name}', np.mean(val_losses), e)
            writer.add_scalar(f'F1_score/val_{loader.name}', np.mean(val_f1_score), e)
        torch.save(net, f'./Results/{exp_name}/Checkpoints/{e}.pt')
        