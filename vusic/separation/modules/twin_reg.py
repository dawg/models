import torch
import torch.nn as nn


__all__ = ["TwinReg"]


class TwinReg(nn.Module):
    def __init__(self, input_size, context_length, sequence_length, debug):
        """
        Desc:
            create a twindecoder

        Args:
            input_size (int): 

            debug (bool): debug mode
        """
        super(RnnEncoder, self).__init__()

        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        self.input_size = input_size

        self.gru = nn.GRUCell(self.input_size, self.input_size)

        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

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
            params["debug"],
        )

    def forward(self, m_enc):
        """
        Desc:
            Forward pass through Twin Net Regularizer.

        Args:
            
        """

        batch_size = m_enc.size()[0]
        sequence_length = m_enc.size()[1]
        
        m_h_dec = torch.zeros(batch_size, self.input_size).to(self.device)
        m_dec = torch.zeros(batch_size, sequence_length, self.input_size).to(
            self.device
        )

        for ts in range(sequence_length-1, -1, -1):
            m_h_dec = self.gru(m_enc[:, ts, :], m_h_dec)
            m_dec[:, t, :] = m_h_dec

        return m_dec

        pass
