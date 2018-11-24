import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class STFT(torch.nn.Module):
    def __init__(self, filter_width: int = 1024, hop_width: int = 512):
        """
        Desc:
            class that implements a torch.nn.Module to provide an STFT with 
        Args:
            filter_width (int): width of the stft filter window
        
            hop_width (int): distance between center of each window
        """
        super(STFT, self).__init__()

        self.filter_width = filter_width
        self.hop_width = hop_width
        self.forward_transform = None

        scale = self.filter_width / self.hop_width
        fourier_basis = np.fft.fft(np.eye(self.filter_width))

        cutoff = int((self.filter_width / 2 + 1))

        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.from_numpy(fourier_basis[:, None, :]).float()
        inverse_basis = torch.from_numpy(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        ).float()

        self.register_buffer("forward_basis", forward_basis)
        self.register_buffer("inverse_basis", inverse_basis)

    def transform(self, data: torch.tensor):
        """
        Desc: 
            STFT of the data
        Args:
            data (torch.tensor): data to be tranformed
        """
        num_batches = data.size(0)
        num_samples = data.size(1)

        self.num_samples = num_samples

        data = data.view(num_batches, 1, num_samples)
        forward_transform = F.conv1d(
            data,
            Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_width,
            padding=self.filter_width,
        )
        cutoff = int((self.filter_width / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase

    def inverse(self, magnitude: torch.tensor, phase: torch.tensor):
        """
        Desc: 
            ISTFT of the magnitude and phase
        Args:
            magnitude (torch.tensor): magnitude values of the freqency representation

            phase (torch.tensor): phase values of the freqency representation
        """
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_width,
            padding=0,
        )
        inverse_transform = inverse_transform[:, :, self.filter_width :]
        inverse_transform = inverse_transform[:, :, : self.num_samples]
        return inverse_transform

    def forward(self, data):
        self.magnitude, self.phase = self.transform(data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction
