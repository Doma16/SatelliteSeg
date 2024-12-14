from Dataset import SatDataset
from Transform import Transform
from model.our_model import WholeModel
from model.unet import UNet, UNetSmall

from utils import read_json_variable, get_save_name
from config import DTYPE, config, VISUALIZE

import torch
from torch.utils.data import DataLoader
import os

import matplotlib.pyplot as plt

def visualize(image, gt):
    nimg = image.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(nimg, cmap='gray')
    axes[0].set_title('Pred')
    axes[0].axis('off')

    if gt is not None:
        ngt = gt.cpu().numpy()
        if ngt.ndim == 3:
            ngt = gt.permute(1,2,0).cpu().numpy()
        axes[1].imshow(ngt, cmap='gray')
        axes[1].set_title('GT')
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def evaluate(model, dataloader, device):
    model.eval()

    total = 0
    precision = 0
    recall = 0
    f1score = 0
    with torch.no_grad():
        for img, gt in dataloader:
            img, gt = img.to(device, dtype=DTYPE), gt.to(device, dtype=DTYPE)

            out = model(img)

            out = torch.clamp(out, min=0, max=1)
            out = torch.round(out, decimals=0)
            
            gt = torch.clamp(gt, min=0, max=1)
            gt = torch.round(gt, decimals=0)
            
            if VISUALIZE:
                visualize(out[0, 0], gt[0, 0])
            
            out = out.flatten()
            gt = gt.flatten()
           
            TP = torch.sum((gt == 1) & (out == 1))
            FP = torch.sum((gt == 1) & (out == 0))
            FN = torch.sum((gt == 0) & (out == 1))

            prec = TP / (TP + FP + 1e-6)
            rec = TP / (TP + FN + 1e-6)
            precision += prec
            recall += rec
            f1score += 2 * (prec * rec) / (prec + rec + 1e-6)
            total += 1

    precision /= total
    recall /= total
    f1score /= total
    return precision, recall, f1score

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = Transform()

    train_path = read_json_variable('paths.json', 'training')
    train_dataset = SatDataset(train_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    #model = WholeModel().to(device, dtype=DTYPE)
    model = UNet(num_classes=1).to(device, dtype=DTYPE)
    load_path = read_json_variable('paths.json', 'save_path')
    load_path = os.path.join(load_path, get_save_name(model, config)+'cv.pth')
    model.load_state_dict(torch.load(load_path, map_location=device))
    # torchinfo.summary(model, input_size=(1, 3, 400, 400))

    precision, recall, f1score = evaluate(model, train_loader, device)
    print(f'Eval, prec: {precision*100:.2f} recall: {recall*100:.2f} f1score: {f1score*100:.2f}')


if __name__ == '__main__':
    main()