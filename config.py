import torch

BATCH_SIZE = 2
SHUFFLE = True
LR = 1e-3
DTYPE = torch.float32
NUM_EPOCHS = 20

config = [BATCH_SIZE, LR, NUM_EPOCHS, SHUFFLE, DTYPE]