import torch
from torchvision.transforms import v2

class Transform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.image_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Lambda(lambda x: x / 255), 
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.gt_transform = v2.Compose([
            v2.Lambda(lambda x: x / x.max()),
            v2.Lambda(lambda x: x[..., None]),
            v2.ToImage(),
        ])
        
        self.flip = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
    def forward(self, image, ground_truth):
        seed = torch.seed()
        torch.manual_seed(seed)
        im = self.flip(self.image_transform(image))
        torch.manual_seed(seed)
        gt = self.flip(self.gt_transform(ground_truth))
        return im, gt
    
class AdapterTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.image_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Resize(size=(416, 416)),       
            v2.Lambda(lambda x: x / 255), 
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.gt_transform = v2.Compose([
            v2.Lambda(lambda x: x[..., None]),
            v2.Resize(size=(416, 416)),
            v2.Lambda(lambda x: x / x.max()),
            v2.ToImage(),
        ])
        
    def forward(self, image, ground_truth):
        breakpoint()
        return self.image_transform(image), self.gt_transform(ground_truth)
    