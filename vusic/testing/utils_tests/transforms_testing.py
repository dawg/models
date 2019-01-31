import os
import torch
from torch.utils.data import Dataset
import PyQt5
import matplotlib.pyplot as plt

import librosa


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
    print(f_sample.shape)
    f_sample = istft.forward(sample)
    print(f_sample.shape)


    # magnitude, phase, ac = stft.forward(Variable(thing))
    # print(f"{magnitude.shape}, {phase.shape}, {ac.shape}")

    # reconstruction = istft.forward(magnitude, phase, ac)
    # print(f"{reconstruction.shape}")

    # f, subplot = plt.subplots(4, sharex=True)

    # subplot[0].set_title("Audio Sample: {}".format(train_ds.filenames[idx]))
    # subplot[0].set_xlim(0, stft_info['win_length'])
    # subplot[0].plot(range(nsamples), torch.t(f_sample).numpy()[0:nsamples])
    # subplot[0].set_ylabel("Mix")
    # subplot[1].plot(
    #     range(window_width), np.transpose(magnitude.detach().numpy()[0, :, :, 0])
    # )
    # subplot[1].set_ylabel("STFT Mix (Magnitude)")
    # subplot[2].plot(
    #     range(window_width), np.transpose(phase.detach().numpy()[0, :, :, 0])
    # )
    # subplot[2].set_ylabel("STFT Mix (Phase)")
    # subplot[3].plot(
    #     range(reconstruction.shape[1]),
    #     torch.t(reconstruction.detach()).numpy()[0:nsamples],
    # )
    # subplot[3].set_ylabel("Reconstructed Mix (Phase)")

    # plt.show()
