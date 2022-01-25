#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HQSS encoder related modules."""

from logging import raiseExceptions
import six

import torch

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from espnet.nets.pytorch_backend.hqss.prenet import Prenet

from espnet.nets.pytorch_backend.tacotron2.cbhg import CBHG
from espnet.nets.pytorch_backend.tacotron2.cbhg import CBHGLoss

def encoder_init(m):
    """Initialize encoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


class SpeakerClassifier(torch.nn.Module):
    """Speaker classifier module for HQSS. Simply FFN with gradient reversal.
    """

    def __init__(
        self,
        idim,
        nspeakers,
        hidden_size=512
    ):
        """Initialize SC module.

        Args:
            idim (int) Dimension of input features (usually num_mels)
            nspeakers (int) number of speakers.
            hidden_size (int, optional) Dimension of FFN hidden size
        """
        super(SpeakerClassifier, self).__init__()
        # store the hyperparameters
        self.nspeakers = nspeakers

        self.attn = torch.nn.MultiheadAttention(idim,4)

        # define network layer modules
        self.ffn = torch.nn.Sequential(
          torch.nn.Linear(idim, hidden_size),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_size, nspeakers),
        )

        self.logsoftmax = torch.nn.LogSoftmax()

    def forward(self, xs):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the padded sequence. Either character ids (B, Tmax)
                or acoustic feature (B, Tmax, idim * encoder_reduction_factor). Padded
                value should be 0.
            ilens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Batch of the sequences of encoder states(B, Tmax, eunits).
            LongTensor: Batch of lengths of each sequence (B,)

        """
        attn_out, _ = self.attn(xs,xs,xs)
        ff_out = self.ffn(attn_out)
        ff_out = torch.sum(ff_out,1)
        return ff_out
    
    def backward(self,grad_output):
      return grad_output.neg()

    def inference(self, x):
        """Inference.

        Args:
            x (Tensor): The sequeunce of character ids (T,)
                    or acoustic feature (T, idim * encoder_reduction_factor).

        Returns:
            Tensor: The sequences of encoder states(T, eunits).

        """
        raise Exception("Should not be used during inference")
