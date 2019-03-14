import torch
import torch.nn as nn


__all__ = ["KelzCnn"]


class KelzCnn(nn.Module):
    def __init__(self, debug, input_features, output_features):
        """
        Desc:
            Create implementation of a CNN stack, the (N,C,H,W) of the dataset is as follows: 
            N = batch size  = x
            C = channels    = initially 3
            H = height      = 626
            W = width       = 226

        Args:
            debug (bool): debug mode
            input_features():
            output_features():
        """
        super(KelzCnn, self).__init__()
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"
        
        self.cnn = nn.Sequential(
            # conv: CxHxw   --> CxHxW
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            # norm: same shape as ^
            nn.BatchNorm2d(output_features // 16),
            # relu: same shape as ^
            nn.ReLU(),
            # conv: 
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            # norm: same shape as ^
            nn.BatchNorm2d(output_features // 16),
            # relu: same shape as ^
            nn.ReLU(),
            # pool: 
            nn.MaxPool2d((1, 2)),
            # drop: same shape as ^
            nn.Dropout(0.25),
            # conv:
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            # norm: same shape as ^
            nn.BatchNorm2d(output_features // 8),
            # relu: same shape as ^
            nn.ReLU(),
            # pool: 
            nn.MaxPool2d((1, 2)),
            # drop: same shape as ^
            nn.Dropout(0.25),
        )

        self.fnn = nn.Sequential(
            # fnn:
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            # drop: same shape as ^
            nn.Dropout(0.5)
        )



    def forward(self, x):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x
