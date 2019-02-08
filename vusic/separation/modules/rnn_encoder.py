import torch
import torch.nn as nn


__all__ = ["RnnEncoder"]


class RnnEncoder(nn.Module):
    def __init__(self, input_size, context_length, sequence_length, debug):
        """
        Desc:
            create an RNN encoder

        Args:
            input_size (int): shape of the input

            context_length (int): the context length

            sequence_length (int): the sequence length

            debug (bool): debug mode
        """
        super(RnnEncoder, self).__init__()

        self.input_size = input_size
        self.context_length = context_length
        self.sequence_length = sequence_length

        # init forward RNN
        self.gru_enc_f = nn.GRUCell(input_size=self.input_size, hidden_size=744)
       
        # init backward RNN
        self.gru_enc_b = nn.GRUCell(input_size=self.input_size, hidden_size=744)

        # train on GPU or CPU
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        # initialize weights and biases
        self.init_w_b()

    def init_w_b(self):
        """
            Desc: 
                initialize weights and biases for the network
        """

        # init input hidden forward weights
        nn.init.xavier_normal_(self.gru_enc_f.weight_ih)

        # init input hidden^2 forward weights
        nn.init.orthogonal_(self.gru_enc_f.weight_hh)

        # init input hidden forward bias
        self.gru_enc_f.bias_ih.data.zero_()

        # init input hidden^2 forward bias
        self.gru_enc_f.bias_hh.data.zero_()

        # init input hidden backward weights
        nn.init.xavier_normal_(self.gru_enc_b.weight_ih)

        # init input hidden^2 backward weights
        nn.init.orthogonal_(self.gru_enc_b.weight_hh)

        # init input hidden backward bias
        self.gru_enc_b.bias_ih.data.zero_()

        # init input hidden^2 backward bias
        self.gru_enc_b.bias_hh.data.zero_()

    @classmethod
    def from_params(cls, params):
        """
        Desc: 
            create an RNN encoder from parameters

        Args:
            param (object): parameters for creating the RNN. Must contain the following
                input_size (int): shape of the input

                context_length (int): length 

                debug (bool): debug mode
        """
        # todo add defaults
        return cls(params["input_size"], params["context_length"], params["sequence_length"], params["debug"])

    def forward(self, windows):
        """
        Desc:
            Forward pass through RNN encoder.

        Args:
            windows (torch.float[sequence_length, window size > ]): Set of batch_size frequency windows.
 
            The output of the RNN encoder of the Masker.
            
        """

        print(f"windows {windows.shape}")

        batch_size = windows.size()[0]
        sequence_length = windows.size()[1]

        forward_t = torch.zeros(batch_size, self.input_size).to(self.device)
        backward_t = torch.zeros(batch_size, self.input_size).to(self.device)

        m_enc = torch.zeros(
            batch_size, sequence_length - (2 * self.context_length), 2 * self.input_size
        ).to(self.device)

        # remove some windows
        windows_reduced = windows[:, :, :self.input_size]

        for t in range(sequence_length):
            
            forward_t = self.gru_enc_f((windows_reduced[:, t, :]), forward_t)
            backward_t = self.gru_enc_b((windows_reduced[:, sequence_length - t - 1, :]), backward_t)

            if self.context_length <= t < sequence_length - self.context_length:
                m_h_enc = torch.cat([
                    forward_t + windows_reduced[:, t, :], 
                    backward_t + windows_reduced[:, sequence_length - t - 1, :]
                    ], dim=1)

                m_enc[:, t - self.context_length, :] = m_h_enc

        return m_enc
