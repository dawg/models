from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os

class SeparationDataset(Dataset):
    def __init__(self, root_dir: str, transform: callable = None, asnp: bool = False):
        """
        Args:
            root_dir (string): Directory with the mix and vocals directories. 
            This directory contains all of the sub folders
        
            transform (callable, optional): Optional transform to be applied 
            on a sample.

            asnp (bool): load as numpy arrays? Will save as pytorch tensor by default
        """

        self.root_dir = root_dir
        self.transform = transform
        self.asnp = asnp

        suffix = ".npy" if asnp else ".pth"

        self.filenames = [
            name
            for name in os.listdir(os.path.join(root_dir, "mix"))
            if name.endswith(suffix)
        ]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):

        mixpath = os.path.join(self.root_dir, "mix", self.filenames[idx])
        vocalpath = os.path.join(self.root_dir, "vocals", self.filenames[idx])

        if self.asnp:
            mix = np.load(mixpath)
            vocals = np.load(vocalpath)
        else:
            # todo how does this compare to GPU?
            mix = torch.load(mixpath).float()
            vocals = torch.load(vocalpath).float()

        sample = {"mix": mix, "vocals": vocals}

        if self.transform:
            sample = self.transform(sample)

        return sample
