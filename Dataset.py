import os
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class SatDataset(Dataset):
    def __init__(self, path, train=True, transform=None, val_transform=None):
        super().__init__()
        self.train = train
        self.image_extension = ".png"
        self.transform = transform
        self.val_transform = val_transform
        self.path = path
        
        self.ground_truth_path = os.path.join(path, "groundtruth")
        self.images_path = os.path.join(path, "images")
        
        n = len(self.ground_truth_path)
        n = int(n * 0.8 if self.train else n * 0.2)
        
        self.images_path = os.listdir(self.images_path)
        self.ground_truth_path = os.listdir(self.ground_truth_path)

        self.images_path = self.images_path[:n] if self.train else self.images_path[-n:] 
        self.ground_truth_path= self.ground_truth_path[:n] if self.train else self.ground_truth_path[-n:] 

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, index):
        
        image = cv2.imread(os.path.join(self.path, "images", self.images_path[index]), cv2.IMREAD_UNCHANGED)
        ground_truth = cv2.imread(os.path.join(self.path, "groundtruth", self.ground_truth_path[index]), cv2.IMREAD_GRAYSCALE)
        
        t = self.transform if self.train else self.val_transform 
        if t:
            image, ground_truth = t(image, ground_truth)
        
        return image, ground_truth
    
if __name__ == '__main__':
    dataset = SatDataset("./data/training", train=True)
    dataset = SatDataset("./data/training", train=False)
    image, ground_truth = dataset.__getitem__(0)
    breakpoint()
    
    fig = plt.figure(figsize=(16, 8))
    ax_image = fig.add_subplot(2, 1, 1)
    ax_gt = fig.add_subplot(2, 1, 2)
	
    ax_image.imshow(image)
    ax_gt.imshow(ground_truth)
    plt.show()