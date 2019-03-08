import torch
import torch.nn as nn


__all__ = ["CnnStack"]


class CnnStack(nn.Module):
    def __init__(self, debug):
        """
        Desc:
            Create implementation of a CNN stack, the (N,C,H,W) of the dataset is as follows: 
            N = batch size  = x
            C = channels    = initially 3
            H = height      = 626
            W = width       = 226

        Args:
            debug (bool): debug mode
        """
        super(CnnStack, self).__init__()
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        # conv: 3*626*226   --> 32*626*226
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1)
        # conv: 32*626*226  --> 32*626*226
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1)
        # norm: 32*626*226  --> 32*626*226
        self.normalization = nn.BatchNorm2d(32)
        # pool: 32*626*226  --> 32*626*113
        self.pooling = nn.MaxPool2d(kernel_size=(1, 2))
        # drop: 32*626*113  --> 32*626*113
        self.dropout1 = nn.Dropout2d(p=0.25)
        # conv: 32*626*113  --> 64*626*113
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        # drop: 64*626*113  --> 64*626*113
        self.dropout2 = nn.Dropout2d(p=0.25)
        # fnn:  64*626*113  --> 512*626*113
        self.fnn = nn.Linear(64, 512)
        # drop: 512*626*113 --> 512*626*113
        self.dropout3 = nn.Dropout2d(p=0.5)

    def forward(self, x):
        print("In forward function")
        print(f"Shape before conv1 {x.shape}")
        out = self.conv1(x)
        print(f"Shape before conv2 {out.shape}")
        out = self.conv2(out)
        print(f"Shape before normalization {out.shape}")
        out = self.normalization(out)
        print(f"Shape before pooling {out.shape}")
        out = self.pooling(out)
        print(f"Shape before dropout1 {out.shape}")
        out = self.dropout1(out)
        print(f"Shape before conv3 {out.shape}")
        out = self.conv3(out)
        print(f"Shape before dropout2 {out.shape}")
        out = self.dropout2(out)
        print(f"Shape before fnn {out.shape}")
        out = self.fnn(out)
        print(f"Shape before dropout3 {out.shape}")
        out = self.dropout3(out)
        print(f"Final shape {out.shape}")
        return out
