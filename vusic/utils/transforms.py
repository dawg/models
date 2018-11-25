from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.signal


class STFT(nn.Module):
    def __init__(
        self, window_size: int = 1024, hop_size: int = 512, window: str = "hanning"
    ):
        """
        Desc: 
            create an STFT object

        Args:
            src (string): directory containing raw samples

            dst (string): destination for binary files

            asnp (bool): save as numpy arrays? Will save as pytorch tensor by default
            
            logger (object, optional): logger
        """

        super(STFT, self).__init__()

        # break if the window_size isn't an even number
        assert window_size % 2 == 0

        self.hop_size = hop_size
        self.n_freq = n_freq = window_size // 2 + 1

        self.real_kernels, self.imag_kernels = _get_stft_kernels(window_size, window)

    def forward(self, sample: torch.float):
        sample = sample.unsqueeze(1)
        sample = sample.unsqueeze(1)

        magn = F.conv2d(sample, self.real_kernels, stride=self.hop_size)
        phase = F.conv2d(sample, self.imag_kernels, stride=self.hop_size)

        magn = magn.permute(0, 2, 1, 3)
        phase = phase.permute(0, 2, 1, 3)

        # complex conjugate
        phase = -1 * phase[:, :, 1:, :]
        ac = magn[:, :, 0, :]
        magn = magn[:, :, 1:, :]
        return magn, phase, ac


def _get_stft_kernels(window_size: int, window: str):

    # break if the window_size isn't an even number
    assert window_size % 2 == 0

    def kernel_fn(freq, time):
        return np.exp(-1j * (2 * np.pi * time * freq) / float(window_size))

    kernels = np.fromfunction(
        kernel_fn, (window_size // 2 + 1, window_size), dtype=np.float64
    )

    if window == "hanning":
        win_cof = scipy.signal.get_window("hanning", window_size)[np.newaxis, :]
    else:
        win_cof = np.ones((1, window_size), dtype=np.float64)

    kernels = kernels[:, np.newaxis, np.newaxis, :] * win_cof

    real_kernels = nn.Parameter(torch.from_numpy(np.real(kernels)).float())
    imag_kernels = nn.Parameter(torch.from_numpy(np.imag(kernels)).float())

    return real_kernels, imag_kernels


class ISTFT(nn.Module):
    def __init__(self, window_size=1024, hop_size=512):
        """
        Desc: 
            create an ISTFT object

        Args:
            window_size (int): the size of the window used for the stft

            hope_length (int): the length of the hop step used in the stft
        """
        super(ISTFT, self).__init__()

        # break if the window_size isn't an even number
        assert window_size % 2 == 0

        # break if the window size is less than the hop size
        assert hop_size <= window_size
        self.hop_size = hop_size

        self.window_size = int(window_size)
        self.n_freq = n_freq = int(window_size / 2)
        self.real_kernels, self.imag_kernels, self.ac_cof = _get_istft_kernels(
            window_size
        )
        trans_kernels = np.zeros((window_size, window_size), np.float64)
        np.fill_diagonal(trans_kernels, np.ones((window_size,), dtype=np.float64))
        # self.win_cof = 1 / scipy.signal.get_window("hanning", window_size)
        # self.win_cof[0] = 0
        # self.win_cof = torch.from_numpy(self.win_cof).float()
        # self.win_cof = nn.Parameter(self.win_cof, requires_grad=False)
        self.trans_kernels = nn.Parameter(
            torch.from_numpy(trans_kernels[:, np.newaxis, np.newaxis, :]).float()
        )

    def forward(self, magn, phase, ac):
        """
        Desc: 
            Implementing the forward method from the nn.module to obtain
            the ISFFT            

        Args:
        """
        assert magn.size()[2] == phase.size()[2] == self.n_freq
        window_size = self.window_size
        hop = self.hop_size

        # complex conjugate
        phase = -1.0 * phase
        real_part = F.conv2d(magn, self.real_kernels)
        imag_part = F.conv2d(phase, self.imag_kernels)

        output = real_part - imag_part

        ac = ac.unsqueeze(1)
        ac = float(self.ac_cof) * ac.expand_as(output)
        output = output + ac
        output = output / float(self.window_size)

        output = F.conv_transpose2d(output, self.trans_kernels, stride=self.hop_size)
        output = output.squeeze(1)
        output = output.squeeze(1)
        # output[:, :hop] = output[:, :hop].mul(self.win_cof[:hop])
        # output[:, -(window_size - hop):] = output[:, -(window_size - hop):].mul(self.win_cof[-(window_size - hop):])
        return output


def _get_istft_kernels(window_size):
    window_size = int(window_size)
    assert window_size % 2 == 0

    def kernel_fn(time, freq):
        return np.exp(1j * (2 * np.pi * time * freq) / window_size)

    kernels = np.fromfunction(
        kernel_fn, (int(window_size), int(window_size / 2 + 1)), dtype=np.float64
    )

    ac_cof = float(np.real(kernels[0, 0]))

    kernels = 2 * kernels[:, 1:]
    kernels[:, -1] = kernels[:, -1] / 2.0

    real_kernels = np.real(kernels)
    imag_kernels = np.imag(kernels)

    real_kernels = nn.Parameter(
        torch.from_numpy(real_kernels[:, np.newaxis, :, np.newaxis]).float()
    )
    imag_kernels = nn.Parameter(
        torch.from_numpy(imag_kernels[:, np.newaxis, :, np.newaxis]).float()
    )
    return real_kernels, imag_kernels, ac_cof