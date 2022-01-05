import six

import torch
import torch.nn.functional as F

class Prenet(torch.nn.Module):
    """Prenet module for decoder of Spectrogram prediction network.
    This is a module of Prenet in the HQSS Spectrogram prediction network.

    """

    def __init__(self, idim, n_layers=2, n_units=256, dropout_rate=0.5):
        """Initialize prenet module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of prenet layers.
            n_units (int, optional): The number of prenet units.

        """
        super(Prenet, self).__init__()
        self.dropout_rate = dropout_rate
        self.prenet = torch.nn.ModuleList()
        for layer in six.moves.range(n_layers):
            n_inputs = idim if layer == 0 else n_units
            self.prenet += [
                torch.nn.Sequential(
                    torch.nn.Linear(att_dim, 1), 
                    torch.nn.ReLU(), 
                    torch.nn.Linear(n_inputs, n_units)
                ) for _ in range(num_dist)
            ]

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of input tensors (B, ..., idim).

        Returns:
            Tensor: Batch of output tensors (B, ..., odim).

        """
        for i in six.moves.range(len(self.prenet)):
            x = F.dropout(self.prenet[i](x), self.dropout_rate)
        return x

