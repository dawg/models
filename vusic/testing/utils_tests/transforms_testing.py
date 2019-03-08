import os
import torch
from torch.utils.data import Dataset
import PyQt5
import matplotlib.pyplot as plt

import numpy as np

import librosa.display

from vusic.utils import STFT, ISTFT, SeparationDataset
from vusic.utils.separation_settings import stft_info


if __name__ == "__main__":
    HOME = os.path.expanduser("~")
    TEST = os.path.join(HOME, "storage", "separation", "pt_f_test")
    TRAIN = os.path.join(HOME, "storage", "separation", "pt_f_train")
    train_ds = SeparationDataset(TRAIN)
    # test_ds = SeparationDataset(TEST)

    # print(f"Training set contains {len(train_ds)} samples.")
    # # print(f"Testing set contains {len(test_ds)} samples.")

    # sample = train_ds[0]

    # f_sample = sample["mix"]

    # D = librosa.amplitude_to_db(f_sample["mg"], ref=np.max)
    # plt.figure(1)
    # librosa.display.specshow(D, y_axis="log")
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("Logarithmic Frequency Power Spectrogram of Mixture")
    # plt.show()

    # D = librosa.amplitude_to_db(sample["vocals"]["mg"], ref=np.max)
    # plt.figure(2)
    # librosa.display.specshow(D, y_axis="log")
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("Logarithmic Frequency Power Spectrogram of ")
    # plt.show()

    x = torch.load("output/masker_loss.pth")    

    print(f"size {len(x)}")
    plt.figure(1)
    plt.plot(x)
    plt.show()
