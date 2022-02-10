#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HQSS encoder related modules."""

import six

import torch

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from espnet.nets.pytorch_backend.hqss.prenet import Prenet

from espnet.nets.pytorch_backend.hqss.cbhg import CBHG

def encoder_init(m):
    """Initialize encoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


class Encoder(torch.nn.Module):
    """Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in HQSS,
    which described in `High Quality Streaming Speech Synthesis with Low, Sentence-Length-Independent Latency`_. This is the encoder which converts either a sequence
    of characters or acoustic features into the sequence of hidden states.

    .. _`High Quality Streaming Speech Synthesis with Low, Sentence-Length-Independent Latency`:
       https://arxiv.org/pdf/2111.09052.pdf

    """

    def __init__(
        self,
        idim,
        input_layer="embed",
        embed_dim=512,
        cbhg_layers=1,
        prenet_layers=1,
        prenet_units=256,
        econv_chans=512,
        use_batch_norm=True,
        dropout_rate=0.5,
        padding_idx=0,
    ):
        """Initialize encoder module.

        Args:
            idim (int) Dimension of the inputs.
            input_layer (str): Input layer type.
            embed_dim (int, optional) Dimension of character embedding.
            cbhg_layers (int, optional) The number of encoder conv layers.
            econv_chans (int, optional) The number of encoder conv filter channels.
            use_batch_norm (bool, optional) Whether to use batch normalization.
            dropout_rate (float, optional) Dropout rate.
        """
        super(Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim

        # define network layer modules
        self.embed = torch.nn.Embedding(idim, embed_dim, padding_idx=padding_idx)

        self.prenet = Prenet(embed_dim, n_layers=prenet_layers, n_units=prenet_units, dropout_rate=dropout_rate)

        self.convs = torch.nn.ModuleList()
        for layer in six.moves.range(cbhg_layers):
            if layer == 0:
                self.convs += [ CBHG(prenet_units, econv_chans) ]
            else:
                self.convs += [ CBHG(econv_chans, econv_chans) ]

        # initialize
        self.apply(encoder_init)

    def forward(self, xs, ilens:torch.Tensor):
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
        
        xs_emb = self.embed(xs)
        
        xs_pre = self.prenet(xs_emb)

        cbhg_out = xs_pre
        
        for conv in self.convs:
            cbhg_out, _ = conv(cbhg_out, ilens)
        
        xs_cbhg = pack_padded_sequence(cbhg_out, ilens, batch_first=True, enforce_sorted=False)
        xs_cbhg, hlens = pad_packed_sequence(xs_cbhg, batch_first=True)

        return xs_cbhg, hlens

    def inference(self, x):
        """Inference.

        Args:
            x (Tensor): The sequeunce of character ids (T,)
                    or acoustic feature (T, idim * encoder_reduction_factor).

        Returns:
            Tensor: The sequences of encoder states(T, eunits).

        """
        xs = x.unsqueeze(0)
        ilens = torch.tensor([x.size(0)])

        return self.forward(xs, ilens)[0][0]
