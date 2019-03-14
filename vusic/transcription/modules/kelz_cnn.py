import torch
import torch.nn as nn


__all__ = ["KelzCnn"]


class KelzCnn(nn.Module):
    def __init__(self, debug=True):
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
        super(KelzCnn, self).__init__()
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"
        
        #TODO Amir transpose this shizzzz TODO

        # conv: 3*226*626   --> 32*224*624
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1)
        # pool(1x2): 32*224*624 --> 32*224*312
        self.pooling1 = nn.MaxPool2d(kernel_size=(1,2))
        # conv: 32*224*312  --> 32*222*310
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1)
        # norm: 32*222*310  --> 32*222*310
        self.normalization = nn.BatchNorm2d(32)
        # pool(2x4): 32*222*310  --> 32*111*77
        self.pooling2 = nn.MaxPool2d(kernel_size=(2,4))
        # drop: 32*111*77  --> 32*111*77
        self.dropout1 = nn.Dropout2d(p=0.25)
        # conv: 32*111*77  --> 64*109*75
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        # pool(2x4): 64*109*75 --> 64*54*18
        self.pooling3 = nn.MaxPool2s(kernel_size=(2,4))
        # drop: 64*54*18  --> 64*54*18
        self.dropout2 = nn.Dropout2d(p=0.25)
        #reshape (flatten) to 62208 (64*54*18)
        # fnn:  62208  --> 512
        self.fnn = nn.Linear(25088, 512)
        # drop: 512 --> 512
        self.dropout3 = nn.Dropout2d(p=0.5)

    def forward(self, x):
        print("In forward function")
        print(f"Shape before conv1 {x.shape}")
        out = self.conv1(x)
        print(f"Shape before pooling1 {out.shape}")
        out = self.pooling1(out)
        print(f"Shape before conv2 {out.shape}")
        out = self.conv2(out)
        print(f"Shape before normalization {out.shape}")
        out = self.normalization(out)
        print(f"Shape before pooling2 {out.shape}")
        out = self.pooling2(out)
        print(f"Shape before dropout1 {out.shape}")
        out = self.dropout1(out)
        print(f"Shape before conv3 {out.shape}")
        out = self.conv3(out)
        print(f"Shape before pooling3 {out.shape}")
        out = self.pooling3(out)
        print(f"Shape before dropout2 {out.shape}")
        out = self.dropout2(out)
        print("Flattening")
        out = out.view(-1, 62208)
        print(f"Shape before fnn {out.shape}")
        out = self.fnn(out)
        print(f"Shape before dropout3 {out.shape}")
        out = self.dropout3(out)
        print(f"Final shape {out.shape}")
        return out
