import torch
import torch.nn as nn

__all__ = ["FnnMasker"]


class FnnMasker(nn.Module):
    def __init__(self, in_dim, out_dim, context_length, debug):
        """
        Desc:
            create a FNN for the Masker

        Args:
            in_dim (int): shape of the input

            out_dim (int): shape of the output

            context_length (int): the context length

            debug (bool): debug mode
        """
        super(FnnMasker, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_length = context_length
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        self.linear_layer = nn.Linear(self._input_dim, self._output_dim)

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
            create an RNN decoder from parameters

        Args:
            param (object): parameters for creating the RNN. Must contain the following
                in_dim (int): shape of the input

                debug (bool): debug mode
        """
        # todo add defaults
        return cls(params["in_dim"], params["debug"])

    def forward(self, h_j_dec, v_in):
        """
        Desc:
            Feed forward through FNN

        Args:
            h_j_dec (torch.autograd.variable.Variable): The output from the RNN decoder

            v_in (numpy.core.multiarray.ndarray): The original magnitude spectrogram input

        Returns:

            The output of the AffineTransform of the masker (torch.autograd.variable.Variable)
        """
        v_in_prime = v_in[:, self._context_length : -self._context_length, :]
        m_j = relu(self.linear_layer(h_j_dec))
        v_j_filt_prime = m_j.mul(v_in_prime)

        return v_j_filt_prime


# EOF
