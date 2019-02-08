import torch
import torch.nn as nn

__all__ = ["FnnMasker"]


class FnnMasker(nn.Module):
    def __init__(self, input_size, output_size, context_length, debug):
        """
        Desc:
            create a FNN for the Masker

        Args:
            input_size (int): shape of the input

            output_size (int): shape of the output

            context_length (int): the context length

            debug (bool): debug mode
        """
        super(FnnMasker, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.context_length = context_length
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        self.linear_layer = nn.Linear(self.input_size, self.output_size)

        self.init_w_b()

    def init_w_b(self):
        """
        Desc: 
            Initialize weights and biases for the network
        """
        #  init input linear weight
        nn.init.xavier_normal_(self.linear_layer.weight)

        # init input linear bias
        self.linear_layer.bias.data.zero_()

    @classmethod
    def from_params(cls, params):
        """
        Desc: 
            create an FNN masker from parameters

        Args:
            param (object): parameters for creating the FNN. See constructor for ingredients
        """
        # todo add defaults
        return cls(params["input_size"], params['output_size'], params['context_length'], params["debug"])

    def forward(self, m_dec, mix_mg_sequence):
        """
        Desc:
            Feed forward through FNN

        Args:
            h_j_dec (torch.autograd.variable.Variable): The output from the RNN decoder

            v_in (numpy.core.multiarray.ndarray): The original magnitude spectrogram input

        Returns:

            The output of the AffineTransform of the masker (torch.autograd.variable.Variable)
        """
        mix_mg_sequence_prime = mix_mg_sequence[:, self.context_length : -self.context_length, :]
        m_j = nn.functional.relu(self.linear_layer(m_dec))
        m_masked = m_j.mul(mix_mg_sequence_prime)

        return m_masked
