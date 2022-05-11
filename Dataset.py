import torch
from torch.utils.data import Dataset, DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x.clone().detach()
        self.y = y.clone().detach()
        self.length = self.x.shape[0]
        
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    
    def __len__(self):
        return self.length 