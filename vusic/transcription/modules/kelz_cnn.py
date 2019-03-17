import torch
import torch.nn as nn


__all__ = ["KelzCnn"]


class KelzCnn(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Desc:
            Create implementation of a CNN stack, the (N,C,H,W) of the dataset is as follows: 
            N = batch size  = 
            C = channels    = 
            H = height      = 
            W = width       = 

        Args:
            debug (bool): debug mode

            in_features ():

            out_features ():
        """
        super().__init__()

        self.cnn = nn.Sequential(
            # Layer 0
            # conv: 1xHxw   --> 48xHxW
            nn.Conv2d(1, out_features // 16, 3, padding=1),
            # norm: same shape as ^
            nn.BatchNorm2d(out_features // 16),
            # relu: same shape as ^
            nn.ReLU(),
            # Layer 1
            # conv:
            nn.Conv2d(out_features // 16, out_features // 16, 3, padding=1),
            # norm: same shape as ^
            nn.BatchNorm2d(out_features // 16),
            # relu: same shape as ^
            nn.ReLU(),
            # Layer 2
            # pool:
            nn.MaxPool2d((1, 2)),
            # drop: same shape as ^
            nn.Dropout(0.25),
            # conv:
            nn.Conv2d(out_features // 16, out_features // 8, 3, padding=1),
            # norm: same shape as ^
            nn.BatchNorm2d(out_features // 8),
            # relu: same shape as ^
            nn.ReLU(),
            # Layer 3
            # pool:
            nn.MaxPool2d((1, 2)),
            # drop: same shape as ^
            nn.Dropout(0.25),
        )

        self.fnn = nn.Sequential(
            # fnn:
            nn.Linear((out_features // 8) * (in_features // 4), out_features),
            # drop: same shape as ^
            nn.Dropout(0.5),
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fnn(x)
        return x
