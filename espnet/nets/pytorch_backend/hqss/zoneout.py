import torch

class ZoneOutCell(torch.nn.Module):
    """ZoneOut Cell module.

    This is a module of zoneout described in
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`_.
    This code is modified from `eladhoffer/seq2seq.pytorch`_.

    Examples:
        >>> lstm = torch.nn.LSTMCell(16, 32)
        >>> lstm = ZoneOutCell(lstm, 0.5)

    .. _`Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`:
        https://arxiv.org/abs/1606.01305

    .. _`eladhoffer/seq2seq.pytorch`:
        https://github.com/eladhoffer/seq2seq.pytorch

    """

    def __init__(self, cell, zoneout_rate=0.1):
        """Initialize zone out cell module.

        Args:
            cell (torch.nn.Module): Pytorch recurrent cell module
                e.g. `torch.nn.Module.LSTMCell`.
            zoneout_rate (float, optional): Probability of zoneout from 0.0 to 1.0.

        """
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_rate = zoneout_rate
        if zoneout_rate > 1.0 or zoneout_rate < 0.0:
            raise ValueError(
                "zoneout probability must be in the range from 0.0 to 1.0."
            )

    def forward(self, inputs, hidden):
        """Calculate forward propagation.

        Args:
            inputs (Tensor): Batch of input tensor (B, input_size).
            hidden (tuple):
                - Tensor: Batch of initial hidden states (B, hidden_size).
                - Tensor: Batch of initial cell states (B, hidden_size).

        Returns:
            tuple:
                - Tensor: Batch of next hidden states (B, hidden_size).
                - Tensor: Batch of next cell states (B, hidden_size).

        """
        #print(inputs.size())
        #print(hidden.size())
        output, next_hidden = self.cell(inputs, hidden)
        if hidden is not None:
          next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_rate)
        return output, next_hidden

    def _zoneout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple(
                [self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)]
            )

        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h

