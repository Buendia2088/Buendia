import torch
import os
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, mode: str = 'train'):
        super(MyDataset, self).__init__()
        
        if mode == 'train':
            self.data = torch.load(os.path.join(data_dir, 'train_data.pt'))
            self.labels = torch.load(os.path.join(data_dir, 'train_labels.pt'))
        elif mode == 'val':
            self.data = torch.load(os.path.join(data_dir, 'val_data.pt'))
            self.labels = torch.load(os.path.join(data_dir, 'val_labels.pt'))
        elif mode == 'test':
            self.data = torch.load(os.path.join(data_dir, 'test_data.pt'))
            self.labels = torch.load(os.path.join(data_dir, 'test_labels.pt'))

        self.transform = transform


    def __getitem__(self, index):

        sample = self.data[index]
        sample_label = self.labels[index]

        if self.transform:
            sample = self.transform(sample)

        sample = sample / (3*28*28)

        return sample, sample_label

    def __len__(self):
        return len(self.data)
