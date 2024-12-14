import os
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class SatDataset(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.image_extension = ".png"
        self.transform = transform
        
        self.ground_truth_path = os.path.join(path, "groundtruth")
        self.images_path = os.path.join(path, "images")
        self.size = len(os.listdir(self.images_path))
        
        assert self.size == len(os.listdir(self.ground_truth_path))
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        index += 1
        zeros = "0" * (3 - len(str(index)))
        
        file_name = "satImage_" + zeros + str(index) + self.image_extension
        
        image = cv2.imread(os.path.join(self.images_path, file_name), cv2.IMREAD_UNCHANGED)
        ground_truth = cv2.imread(os.path.join(self.ground_truth_path, file_name), cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            image, ground_truth = self.transform(image, ground_truth)
        
        return image, ground_truth
    
if __name__ == '__main__':
    dataset = SatDataset("./data/training")
    image, ground_truth = dataset.__getitem__(0)
    
    fig = plt.figure(figsize=(16, 8))
    ax_image = fig.add_subplot(3, 1, 1)
    ax_gt = fig.add_subplot(3, 1, 2)
	
    ax_image.imshow(image)
    ax_gt.imshow(ground_truth)
    plt.show()