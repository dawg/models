import torch
import torch.nn as nn

__all__ = ["AttentionMechanism"]

class AttentionMechanism(nn.Module):
    def __init__(self, in_dim, debug):
        """
        Desc:
            create an attention mechanism for the Masker

        Args:
            in_dim (int): shape of the input

            debug (bool): debug mode
        """
        super(AttentionMechanism, self).__init__()

        # output dim
        self.linear_out = nn.Linear(in_dim*2, in_dim)
        self.device = "cuda" if not debug and torch.cuda.is_available() else "cpu"

        
    def set_mask(self, mask):
        """
        Desc:
            sets indices to be masked

        Args:
            mask (torch.Tensor): tensor with indices to be masked
        """
        # Tensor with indices to be masked 
        self.mask = mask

    def forward(self, output, context):
        """
        Desc:
            TODO this whole section needs work
        
        Args:
            Arguments
        """

        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = nn.functional.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = nn.functional.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn