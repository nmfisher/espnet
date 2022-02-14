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


class ProsodyEncoder(torch.nn.Module):
    """Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in HQSS,
    which described in `High Quality Streaming Speech Synthesis with Low, Sentence-Length-Independent Latency`_. This is the encoder which converts either a sequence
    of characters or acoustic features into the sequence of hidden states.

    .. _`High Quality Streaming Speech Synthesis with Low, Sentence-Length-Independent Latency`:
       https://arxiv.org/pdf/2111.09052.pdf

    """

    def __init__(
        self,
        embed_dim=128,
        cbhg_layers=1,
        prenet_layers=2,
        prenet_units=256,
        econv_chans=512,
        use_batch_norm=True,
        dropout_rate=0.5,
        num_clusters=15,
        padding_idx=0,
    ):
        """Initialize prosody encoder module.

        Args:
            idim (int) Dimension of the inputs.
            input_layer (str): Input layer type.
            embed_dim (int, optional) Dimension of character embedding.
            cbhg_layers (int, optional) The number of encoder conv layers.
            econv_chans (int, optional) The number of encoder conv filter channels.
            use_batch_norm (bool, optional) Whether to use batch normalization.
            dropout_rate (float, optional) Dropout rate.
        """
        super(ProsodyEncoder, self).__init__()

        # define network layer modules
        self.pitch_embed = torch.nn.Embedding(num_clusters, embed_dim, padding_idx=padding_idx)
        self.duration_embed = torch.nn.Embedding(num_clusters, embed_dim,  padding_idx=padding_idx)

        self.prenet = Prenet(embed_dim*2, n_layers=prenet_layers, n_units=prenet_units)

        self.convs = torch.nn.ModuleList()
        for layer in six.moves.range(cbhg_layers):
            if layer == 0:
                self.convs += [ CBHG(prenet_units, econv_chans) ]
            else:
                self.convs += [ CBHG(econv_chans, econv_chans) ]

        # initialize
        self.apply(encoder_init)

    def forward(self, durations, pitch, ilens:torch.Tensor):
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
        durations_emb = self.duration_embed(durations)
        pitch_emb = self.pitch_embed(pitch)
        
        xs_emb = torch.cat([durations_emb, pitch_emb],-1)

        xs_pre = self.prenet(xs_emb)

        cbhg_out = xs_pre
        
        for conv in self.convs:
            cbhg_out, _ = conv(cbhg_out, ilens) 

        if self.training:
          xs_cbhg = pack_padded_sequence(cbhg_out, ilens.cpu(), batch_first=True, enforce_sorted=False)
          xs_cbhg, hlens = pad_packed_sequence(xs_cbhg, batch_first=True)
        else:
          xs_cbhg = cbhg_out
          hlens = None

        return xs_cbhg, hlens
        