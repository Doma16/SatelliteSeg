import random
import torch
from Dataset import SatDataset
from Transform import Transform

from config import BATCH_SIZE

def cross_validation(n_splits=5):
    transform = Transform()
    dataset = SatDataset("./data/training", transform)
    
    ids = list(range(len(dataset)))
    split_size = len(dataset) // n_splits
    
    random.seed(42)
    random.shuffle(ids)

    loaders = []

    for split_num in range(n_splits):
        test_start = split_size * split_num
        test_end = test_start + split_size
        
        train_ids = ids[:test_start] + ids[test_end:]
        test_ids = ids[test_start:test_end]
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, sampler=train_subsampler
        )
        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, sampler=test_subsampler
        )

        loaders.append((trainloader, testloader))
        
    return loaders