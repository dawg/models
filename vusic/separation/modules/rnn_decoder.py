import torch
import torch.nn as nn

__all__ = ["RnnDecoder"]


class RnnDecoder(nn.Module):
    def __init__(self, in_dim, debug):
        """
        Desc: 
            create an RNN decoder

        Args:
            in_dim (int): shape of the input

            debug (bool): debug mode
        """
        super(RnnDecoder, self).__init__()

        self.in_dim = in_dim
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        # todo: make this rectangular as opposed to square?
        # create gated recurrent unit cells in the shape of our input
        self.gru = nn.GRUCell(self.in_dim, self.in_dim)

        self.init_w_b()

    def init_w_b(self):
        """
            Desc: 
                initialize weights and biases for the network
        """

        # init input hidden weights
        nn.init.xavier_normal_(self.gru.weight_ih)

        # init hidden^2 weights
        nn.init.orthogonal_(self.gru.weight_hh)

        # init input hidden bias
        self.gru.bias_ih.data.zero_()

        # init hidden^2 bias
        self.gru.bias_hh.data.zero_()

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

    def forward(self, encoder_out):
        """
        Desc: 
            feed forward through RNN decoder

        Args:
            encoder_out: output of the RNN encoder
        """
