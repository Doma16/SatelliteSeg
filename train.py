from CrossValidation import cross_validation
from model.our_model import WholeModel
from model.unet import UNet, UNetSmall
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.optim as optim

from utils import read_json_variable, get_save_name
from eval import evaluate

import os
import json
from tqdm import tqdm
from collections import defaultdict
from config import LR, NUM_EPOCHS, DTYPE, config

from losses import BinaryCrossEntropyLoss, BCEDiceLoss

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
    loaders = cross_validation()[:1]
    criterion = BCEDiceLoss()

    precision, recall, f1score = [], [], []
    f1_per_epoch = defaultdict(list)
    for j, (train_loader, test_loader) in enumerate(loaders):

        # Initialize the model
        load_path = read_json_variable('paths.json', 'load_path')
        model = UNet()
        if load_path:
            model.load_state_dict(torch.load(load_path, map_location=device), strict=False)
        model = model.to(device, dtype=DTYPE)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        num_epochs = NUM_EPOCHS
        for epoch in tqdm(range(num_epochs)):
            loss = train(model, train_loader, criterion, optimizer, device)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
            prec, rec, f1 = evaluate(model, test_loader, device)
            f1_per_epoch[j].append(f1)
            print(f'Metrics: prec: {prec*100:.2f} rec: {rec*100:.2f} f1: {f1*100:.2f}')

        save_path = read_json_variable('paths.json', 'save_path')
        save_path = os.path.join(save_path, get_save_name(model, config)+'cv.pth')
        torch.save(model.state_dict(), save_path)
        
        precision.append(prec)
        recall.append(rec)
        f1score.append(f1)

    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    f1score = sum(f1score) / len(f1score)

    final_f1_per_epoch = []
    for v in zip(*f1_per_epoch.values()):
        final_f1_per_epoch.append((sum(v) / len(v)).item())
    
    save_f1_path = read_json_variable('paths.json', 'save_path')
    save_f1_name = get_save_name(model, config) + 'f1_score.json'
    with open(os.path.join(save_f1_path, save_f1_name), 'w') as f:
        json.dump(final_f1_per_epoch, f, indent=4)
    print(f'[CrossValidation] P: {precision*100:.2f} R: {recall*100:.2f} F1: {f1score*100:.2f}')

if __name__ == '__main__':
    main()