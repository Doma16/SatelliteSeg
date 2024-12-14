from Dataset import SatDataset
from Transform import Transform, AdapterTransform
from model.our_model import WholeModel, Adapter
from model.unet import UNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from losses import (
    MSELoss,
    BinaryCrossEntropyLoss,
    BCEDiceLoss,
    IoULoss,
)
from utils import read_json_variable, get_save_name
from eval import evaluate

import os
from tqdm import tqdm
from config import LR, NUM_EPOCHS, DTYPE, config, SHUFFLE, BATCH_SIZE

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for img, gt in dataloader:
        img, gt = img.to(device, dtype=DTYPE), gt.to(device, dtype=DTYPE)

        #forward
        out = model(img)
        loss = criterion(out, gt)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = AdapterTransform()
    dataset = SatDataset(path=read_json_variable('paths.json', 'training'), transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    model = Adapter().to(device, dtype=DTYPE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = BCEDiceLoss()

    num_epochs = NUM_EPOCHS
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        loss = train(model, loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
        prec, rec, f1 = evaluate(model, loader, device)
        pbar.set_description(f'Loss: {loss}')
        print(f'[Eval] P: {prec*100:.2f} R: {rec*100:.2f} F1: {f1*100:.2f}')
        
    save_path = read_json_variable('paths.json', 'save_path')
    save_path = os.path.join(save_path, get_save_name(model, config)+'_end.pth')
    torch.save(model.state_dict(), save_path)    

if __name__ == '__main__':
    main()