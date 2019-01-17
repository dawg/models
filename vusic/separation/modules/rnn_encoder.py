import torch
import torch.nn as nn


__all__ = ["RnnEncoder"]


class RnnEncoder(nn.Module):
    def __init__(self, in_dim, context_length, debug):
        """
        Desc:
            create an RNN encoder

        Args:
            in_dim (int): shape of the input
            context_length (int): the context length
            debug (bool): debug mode
        """
        super(RnnEncoder, self).__init__()

        self.in_dim = in_dim
        self.context_length = context_length

        self.gru_enc_f = nn.GRUCell(self.in_dim, self.in_dim)
        self.gru_enc_b = nn.GRUCell(self.in_dim, self.in_dim)

        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

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

    def forward(self, v_in):
        """Forward pass.
        :param v_in: The input to the RNN encoder of the Masker.
        :type v_in: numpy.core.multiarray.ndarray
        :return: The output of the RNN encoder of the Masker.
        :rtype: torch.autograd.variable.Variable
        """
        batch_size = v_in.size()[0]
        seq_length = v_in.size()[1]

        h_t_f = torch.zeros(batch_size, self._input_dim).to(self._device)
        h_t_b = torch.zeros(batch_size, self._input_dim).to(self._device)

        h_enc = torch.zeros(
            batch_size, seq_length - (2 * self._context_length), 2 * self._input_dim
        ).to(self._device)

        v_tr = v_in[:, :, : self._input_dim]

        for t in range(seq_length):
            h_t_f = self.gru_enc_f((v_tr[:, t, :]), h_t_f)
            h_t_b = self.gru_enc_b((v_tr[:, seq_length - t - 1, :]), h_t_b)

            if self._context_length <= t < seq_length - self._context_length:
                h_t = torch.cat(
                    [h_t_f + v_tr[:, t, :], h_t_b + v_tr[:, seq_length - t - 1, :]],
                    dim=1,
                )
                h_enc[:, t - self._context_length, :] = h_t

        return h_enc

    @classmethod
    def from_params(cls, params):
        """
        Desc: 
            create an RNN encoder from parameters

        Args:
            param (object): parameters for creating the RNN. Must contain the following
                in_dim (int): shape of the input

                debug (bool): debug mode
        """
        # todo add defaults
        return cls(params["in_dim"], params["debug"])
