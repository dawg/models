import torch
import torch.nn as nn

__all__ = ["RnnDecoder"]


class RnnDecoder(nn.Module):
    def __init__(self, input_size, debug):
        """
        Desc: 
            create an RNN decoder

        Args:
            input_size (int): shape of the input

            debug (bool): debug mode
        """
        super(RnnDecoder, self).__init__()

        self.input_size = input_size

        # todo: make this rectangular as opposed to square?
        # create gated recurrent unit cells in the shape of our input
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
            create an RNN decoder from parameters

        Args:
            param (object): parameters for creating the RNN. Must contain the following
                input_size (int): shape of the input

                debug (bool): debug mode
        """
        return cls(params["input_size"], params["debug"])

    def forward(self, m_enc):
        """
        Desc: 
            feed forward through RNN decoder

        Args:
            encoder_out (torch.autograd.variable.Variable): output of the RNN encoder
        """

        batch_size = m_enc.size()[0]
        seq_length = encoder_out.size()[1]
        h_t_dec = torch.zeros(batch_size, self.input_size).to(self.device)
        h_j_dec = torch.zeros(batch_size, seq_length, self.input_size).to(self.device)

        for ts in range(seq_length):
            h_t_dec = self.gru(encoder_out[:, ts, :], h_t_dec)
            m_enc[:, ts, :] = m_enc

        return m_enc
