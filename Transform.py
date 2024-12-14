import torch
from torchvision.transforms import v2

class Transform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.image_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32),
            v2.Lambda(lambda x: x / 255), 
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.gt_transform = v2.Compose([
            v2.Lambda(lambda x: x / x.max()),
            v2.Lambda(lambda x: x[..., None]),
            v2.ToImage(),
        ])
        
    def forward(self, image, ground_truth):
        return self.image_transform(image), self.gt_transform(ground_truth)