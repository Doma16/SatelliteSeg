import torch
from torchvision.transforms import v2
import numpy as np

from config import DTYPE

class Transform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.image_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(DTYPE),
            v2.Lambda(lambda x: x / 255), 
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.Resize(size=(416, 416), antialias=True)
        ])
        self.gt_transform = v2.Compose([
            v2.Lambda(lambda x: np.floor(x / x.max())),
            v2.Lambda(lambda x: x[..., None]),
            v2.ToImage(),
            v2.Resize(size=(416, 416), antialias=True)
        ])
        
        self.flip = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])

        self.rotate = v2.Compose([
            v2.RandomRotation(degrees=(-180, 180)),
            v2.CenterCrop(size=(288, 288))
        ])

        # self.permute = v2.Compose([
        #     v2.PermuteChannels()
        # ])


    def forward(self, image, ground_truth):
        seed = torch.seed()
        torch.manual_seed(seed)
        im = self.flip(self.image_transform(image))
        im = self.rotate(im)
        torch.manual_seed(seed)
        gt = self.flip(self.gt_transform(ground_truth))
        gt = self.rotate(gt)
        # import matplotlib.pyplot as plt
        # tmp = im.permute(1,2,0).cpu().numpy()
        # plt.imshow(tmp)
        # plt.show()
        # breakpoint()
        return im, gt
    
class EvalTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.image_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(DTYPE),
            v2.Lambda(lambda x: x / 255),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.Resize(size=(416, 416), antialias=True)
        ])
        self.gt_transform = v2.Compose([
            v2.Lambda(lambda x: x / x.max()),
            v2.Lambda(lambda x: x[..., None]),
            v2.ToImage(),
            v2.Resize(size=(416, 416), antialias=True)
        ])

    def forward(self, image, ground_truth):
        im = self.image_transform(image)
        gt = self.gt_transform(ground_truth)
        return im, gt

class AdapterTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.image_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(DTYPE),
            v2.Resize(size=(416, 416)),       
            v2.Lambda(lambda x: x / 255), 
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.gt_transform = v2.Compose([
            v2.Lambda(lambda x: x[..., None]),
            v2.Lambda(lambda x: x / x.max()),
            v2.ToImage(),
            v2.Resize(size=(416, 416)),
        ])
        
    def forward(self, image, ground_truth):
        return self.image_transform(image), self.gt_transform(ground_truth)
    