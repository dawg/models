import torch
import torch.nn as nn
from vusic.transcription.modules.keltz_cnn import KelzCnn
from vusic.transcription.modules.bilstm import BiLstm


__all__ = ["Model"]


class Model(nn.Module):
    def __init__(self, debug):
        """
        Desc:

        Args:
            debug (bool): debug mode
        """
        super(Model, self).__init__()
        self.keltz = KelzCnn()
        self.bilstm = BiLstm()
        self.fnn = nn.Linear(512, 88)
        self.sigmoid = nn.LogSigmoid()

    def forward(self, x, y=None):
        if y is None:
            out = self.keltz(x)
            out = self.bilstm(out)
        else:
            out = self.keltz(x)
            out = self.fnn(x)
            concatenated = torch.cat((out, y), 0)
            out = self.bilstm(concatenated)
            out = self.fnn(out)
            out = self.sigmoid(out)
        return out
