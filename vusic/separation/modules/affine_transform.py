import torch
import torch.nn as nn

__all__ = ["AffineTransform"]


class AffineTransform(nn.Module):
    def __init__(self, input_size, debug):
        """
        Desc: 
            create an RNN decoder

        Args:
            input_size (int): shape of the input

            debug (bool): debug mode
        """
        super(AffineTransform, self).__init__()
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.linear_layer = nn.Linear(self.input_size, self.input_size)
        self.init_w_b()

    def init_w_b(self):
        """
            Desc: 
                initialize weights and biases for the network
        """

        # init weights
        nn.init.xavier_normal_(self.linear_layer.weight)

        # init bias
        self.linear_layer.bias.data.zero_()

    @classmethod
    def from_params(cls, params):
        """
        Desc: 
            create from parameters

        Args:
            param (object): parameters for creating the NN. Must contain the following
                input_size (int): shape of the input

                debug (bool): debug mode
        """
        return cls(params["input_size"], params["debug"])

    def forward(self, x):
        """
        Desc: 
            feed forward

        Args:

        """

        return self.linear_layer(x)
