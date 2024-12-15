from model.unet import  SMPUNET

import torch
import torch.optim as optim

from utils import read_json_variable, get_save_name
from eval import evaluate

import os
import json
from tqdm import tqdm
from collections import defaultdict
from config import LR, NUM_EPOCHS, DTYPE, config, BATCH_SIZE, SHUFFLE
from Dataset import SatDataset
from Transform import Transform, EvalTransform
import torch.nn as nn
from losses import BCEDiceLoss, BinaryCrossEntropyLoss, IoULoss
from torch.utils.data import DataLoader

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for img, gt in dataloader:
        img, gt = img.to(device, dtype=DTYPE), gt.to(device, dtype=DTYPE)

        #forward
        out = model(img)
        loss = criterion(out, gt)
        # loss += nn.BCELoss()(out, gt)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = read_json_variable("paths.json", "training")

    transform, val_transform = Transform(), EvalTransform()

    train_ds = SatDataset(path=path, train=True, transform=transform, val_transform=val_transform)
    val_ds = SatDataset(path=path, train=False, transform=transform, val_transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    test_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    criterion = BinaryCrossEntropyLoss()

    per_epoch = defaultdict(list)

    # Initialize the model
    model = SMPUNET()
    model = model.to(device, dtype=DTYPE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    num_epochs = NUM_EPOCHS
    for epoch in tqdm(range(num_epochs)):
        loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
        prec, rec, f1 = evaluate(model, test_loader, device)
        per_epoch['f1'].append(f1.item())
        per_epoch['loss'].append(loss)
        print(f'Metrics: prec: {prec*100:.2f} rec: {rec*100:.2f} f1: {f1*100:.2f}')

    save_path = read_json_variable('paths.json', 'save_path')
    save_path = os.path.join(save_path, get_save_name(model, config)+'.pth')
    torch.save(model.state_dict(), save_path)
    
    save_f1_path = read_json_variable('paths.json', 'save_path')
    save_f1_name = get_save_name(model, config) + '_score.json'
    with open(os.path.join(save_f1_path, save_f1_name), 'w') as f:
        json.dump(dict(per_epoch), f, indent=4)

if __name__ == '__main__':
    main()