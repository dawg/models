# from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# TODO this should inherit from either tensorflow or pytorch's dataset
class SeparationDataset:
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with the mix and vocals directories. 
            This directory contains all of the sub folders
        
            transform (callable, optional): Optional transform to be applied 
            on a sample.
        """
        self.root_dir = root_dir
        self.filenames = [
            name
            for name in os.listdir(os.path.join(root_dir, "mix"))
            if name.endswith(".stem.mp4.npy")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        mix = np.load(os.path.join(self.root_dir, "mix", self.filenames[idx]))
        vocals = np.load(os.path.join(self.root_dir, "vocals", self.filenames[idx]))

        sample = {"mix": mix, "vocals": vocals}

        if self.transform:
            sample = self.transform(sample)

        return sample
