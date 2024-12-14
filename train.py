from CrossValidation import cross_validation
from model.our_model import WholeModel
from model.unet import UNet, UNetSmall

import torch
import torch.nn as nn
import torch.optim as optim

from utils import read_json_variable, get_save_name
from eval import evaluate

import os
from tqdm import tqdm
from config import LR, NUM_EPOCHS, DTYPE, config

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
    loaders = cross_validation()
    criterion = nn.MSELoss()

    precision, recall, f1score = [], [], []
    for train_loader, test_loader in loaders:

        model = WholeModel().to(device, dtype=DTYPE)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        num_epochs = NUM_EPOCHS
        for epoch in tqdm(range(num_epochs)):
            loss = train(model, train_loader, criterion, optimizer, device)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

        save_path = read_json_variable('paths.json', 'save_path')
        save_path = os.path.join(save_path, get_save_name(model, config)+'cv.pth')
        torch.save(model.state_dict(), save_path)
        
        prec, rec, f1 = evaluate(model, test_loader, device)
        precision.append(prec)
        recall.append(rec)
        f1score.append(f1)

    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    f1score = sum(f1score) / len(f1score)

    print(f'[CrossValidation] P: {precision*100:.2f} R: {recall*100:.2f} F1: {f1score*100:.2f}')

if __name__ == '__main__':
    main()