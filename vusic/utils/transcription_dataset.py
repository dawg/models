from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
from vusic.utils.separation_settings import debug


class TranscriptionDataset(Dataset):
    def __init__(self, root_dir: str, transform: callable = None):
        """
        Args:
            root_dir (string): Directory with the mix and vocals directories. 
            This directory contains all of the sub folders
        
            transform (callable, optional): Optional transform to be applied 
            on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform

        suffix = ".pth"

        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        self.filenames = [
            name
            for name in os.listdir(os.path.join(root_dir, "mix"))
            if name.endswith(suffix)
        ]

    @classmethod
    def from_params(cls, params: object):
        """
        Desc: 
            create a SeparationDataset from parameters

        Args:
            param (object): parameters for creating the SeparationDataset. Must contain the following
                root_dir (str): root directory of the dataset

                transform (optional, str): transform to be applied to each sample upon retrieval
        """

        transform = params["transform"] if "transform" in params else None

        return cls(params["dataset"], params["transform"])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):

        mixpath = os.path.join(self.root_dir, "mix", self.filenames[idx])
        vocalpath = os.path.join(self.root_dir, "vocals", self.filenames[idx])

        mix = torch.load(mixpath)
        vocals = torch.load(vocalpath)

        if debug:
            mix["mg"] = mix["mg"].type(torch.float)
            mix["ph"] = mix["ph"].type(torch.float)
            vocals["mg"] = vocals["mg"].type(torch.float)
            vocals["ph"] = vocals["ph"].type(torch.float)

        sample = {"mix": mix, "vocals": vocals}

        if self.transform:
            sample = self.transform(sample)

        return sample
