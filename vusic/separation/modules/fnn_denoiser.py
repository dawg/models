import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FnnMasker"]


class FnnDenoiser(nn.Module):
    def __init__(self, input_size, debug):
        """
        Desc:
            create an FNN for the Denoiser

        Args:
            input_size (int): shape of the input

            debug (bool): debug mode
        """
        super(FnnDenoiser, self).__init__()

        self.input_size = input_size

        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        self.fnn_enc = nn.Linear(self.input_size, int(self.input_size / 2))
        self.fnn_dec = nn.Linear(int(self.input_size / 2), self.input_size)

        self.init_w_b()

    def init_w_b(self):
        """
        Desc: 
            Initialize weights and biases for the network
        """
        nn.init.xavier_normal_(self.fnn_enc.weight)
        self.fnn_enc.bias.data.zero_()
        nn.init.xavier_normal_(self.fnn_dec.weight)
        self.fnn_dec.bias.data.zero_()

    @classmethod
    def from_params(cls, params):
        """
        Desc: 
            create an FNN masker from parameters

        Args:
            param (object): parameters for creating the FNN. See constructor for ingredients
        """
        # todo add defaults
        return cls(params["input_size"], params["debug"])

    def forward(self, m_masked):
        """
        Desc:

        Args:

        Returns:
        """
        fnn_enc_output = F.relu(self.fnn_enc(m_masked))
        fnn_dec_output = F.relu(self.fnn_dec(fnn_enc_output))

        denoised = fnn_dec_output.mul(m_masked)

        return denoised
