import torch
import torch.nn as nn


__all__ = ["TwinReg"]


class TwinReg(nn.Module):
    def __init__(self, input_size, context_length, sequence_length, debug):
        """
        Desc:
            create an RNN encoder

        Args:
            input_size (int): 

            context_length (int): the context length

            sequence_length (int): the sequence length

            debug (bool): debug mode
        """
        super(RnnEncoder, self).__init__()


        # train on GPU or CPU
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"


    def init_w_b(self):
        """
            Desc: 
                initialize weights and biases for the network
        """
        pass

    @classmethod
    def from_params(cls, params):
        """
        Desc: 
            create Twin Net Regularizer from parameters

        Args:
            param (object): parameters for creating the object. Must contain the following
                input_size (int): shape of the input

                context_length (int): length 

                debug (bool): debug mode
        """
        # todo add defaults
        return cls(
            params["input_size"],
            params["context_length"],
            params["sequence_length"],
            params["debug"],
        )

    def forward(self, layers):
        """
        Desc:
            Forward pass through Twin Net Regularizer.

        Args:
            
        """

        pass
