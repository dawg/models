import torch
import torch.nn as nn


__all__ = ["CnnStack"]


class CnnStack(nn.Module):
    def __init__(self, debug):
        """
        Desc:
            Create implementation of a CNN stack, the (N,C,H,W) of the dataset is as follows: 
            N = batch size  =
            C = channels    =
            H = height      =
            W = width       =

        Args:
            debug (bool): debug mode
        """
        super(CnnStack, self).__init__()
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        self.inputlayer = nn.Linear(
            in_features, out_features
        )  # TODO, is this layer necessary?
        self.conv1 = nn.Conv2d(inchannel, 32, 3, stride=1)  # TODO, input channels
        self.conv2 = nn.Conv2d(inchannel, 32, 3, stride=1)  # TODO, input channels
        self.normalization = nn.BatchNorm2d(C)  # TODO C from NCHW
        self.pooling = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.conv3 = nn.Conv2d(inchannel, 64, 3, stride=1)  # TODO, input channels
        self.dropout2 = nn.Dropout2d(p=0.25)
        self.fnn = nn.Linear(64, 512)  # TODO maybe change infeatures
        self.dropout3 = nn.Dropout2d(p=0.5)

    def forward(self, x):
        out = self.inputlayer(x)  # TODO maybe remove layer
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.normalization(out)
        out = self.pooling(out)
        out = self.dropout1(out)
        out = self.conv3(out)
        out = self.dropout2(out)
        out = self.fnn(out)
        out = self.dropout3(out)
        return out
