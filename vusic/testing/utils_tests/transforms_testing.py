import os
import torch
from torch.utils.data import Dataset
import PyQt5
import matplotlib.pyplot as plt

import numpy as np

import librosa.display


from vusic.utils import STFT, ISTFT
from vusic.utils.separation_settings import stft_info

class RawDataset(Dataset):
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

if __name__ == "__main__":
    HOME = os.path.expanduser("~")
    TEST = os.path.join(HOME, "storage", "separation", "pt_test")
    TRAIN = os.path.join(HOME, "storage", "separation", "pt_train")
    train_ds = RawDataset(TRAIN)
    test_ds = RawDataset(TEST)

    print(f"Training set contains {len(train_ds)} samples.")
    print(f"Testing set contains {len(test_ds)} samples.")

    sample = train_ds[0]
    nsamples = 50000

    print(f"{sample['mix'].shape}")
    sample = torch.t(sample["mix"][0,:,:])
    sample = librosa.core.to_mono(sample.numpy())
    print(f"{sample.shape}")

    stft = STFT.from_params(stft_info)
    istft = ISTFT.from_params(stft_info)

    f_sample = stft.forward(sample)
    print(f"{f_sample.shape}")
    sample = istft.forward(f_sample)
    print(f"{sample.shape}")


    D = librosa.amplitude_to_db(np.abs(f_sample), ref=np.max)
    plt.figure(1)
    librosa.display.specshow(D, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Logarithmic Frequency Power Spectrogram')
    plt.show()