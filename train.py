from Dataset import SatDataset
from Transform import Transform
from model.our_model import WholeModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import read_json_variable, get_save_name

import os

from config import BATCH_SIZE, SHUFFLE, LR, NUM_EPOCHS, DTYPE, config

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

    transform = Transform()
    train_path = read_json_variable('paths.json', 'training')
    train_dataset = SatDataset(train_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    model = WholeModel().to(device, dtype=DTYPE)
    # torchinfo.summary(model, input_size=(1, 3, 400, 400))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    num_epochs = NUM_EPOCHS
    for epoch in range(num_epochs):
        loss = train(model, train_loader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

    save_path = read_json_variable('paths.json', 'save_path')
    save_path = os.path.join(save_path, get_save_name(model, config)+'.pth')
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()