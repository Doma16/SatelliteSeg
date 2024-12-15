import torch
import torch.nn as nn
import torch.nn.functional as F
from model.unet import UNetSmall


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, gt):
        return F.mse_loss(pred, gt, reduction='mean')


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, pred, gt):
        return F.binary_cross_entropy_with_logits(pred, gt)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-10):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth  # zbog dijeljenja s nulon

    def forward(self, pred, gt):
        bce_loss = F.binary_cross_entropy_with_logits(pred, gt)
        pred = torch.sigmoid(pred)
        intersection = (pred * gt).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
        dice_loss = 1 - (2. * intersection + self.smooth) / (union + self.smooth)

        total_loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss.mean()

        return total_loss


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-10):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        intersection = (pred * gt).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou.mean()


if __name__ == '__main__':
    N = 2
    N, C, H, W = 1, 1, 3, 3
    pred = torch.ones((N, C, H, W))
    gt = torch.triu(torch.ones((N, C, H, W)))
    print(pred)
    print(gt)
    criterion = MSELoss()
    print(f'MSE={criterion(pred, gt)}')
    criterion = IoULoss()
    print(f'IoU={criterion(pred, gt)}')
    criterion = BCEDiceLoss()
    print(f'BCEDice={criterion(pred, gt)}')
    criterion = BinaryCrossEntropyLoss()
    print(f'BCE={criterion(pred, gt)}')
