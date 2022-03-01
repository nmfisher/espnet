#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron2 encoder related modules."""

import six

import torch

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from typing import Optional


def encoder_init(m):
    """Initialize encoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))


class Encoder(torch.nn.Module):
    """Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in Tacotron2,
    which described in `Natural TTS Synthesis by Conditioning WaveNet on Mel
    Spectrogram Predictions`_. This is the encoder which converts either a sequence
    of characters or acoustic features into the sequence of hidden states.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
        self,
        idim,
        input_layer="embed",
        embed_dim=512,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_chans=512,
        econv_filts=5,
        use_batch_norm=True,
        use_residual=False,
        dropout_rate=0.5,
        padding_idx=0,
        num_prosody_clusters=5,
        prosody_embed_dim=64
    ):
        """Initialize Tacotron2 encoder module.

        Args:
            idim (int) Dimension of the inputs.
            input_layer (str): Input layer type.
            embed_dim (int, optional) Dimension of character embedding.
            elayers (int, optional) The number of encoder blstm layers.
            eunits (int, optional) The number of encoder blstm units.
            econv_layers (int, optional) The number of encoder conv layers.
            econv_filts (int, optional) The number of encoder conv filter size.
            econv_chans (int, optional) The number of encoder conv filter channels.
            use_batch_norm (bool, optional) Whether to use batch normalization.
            use_residual (bool, optional) Whether to use residual connection.
            dropout_rate (float, optional) Dropout rate.

        """
        super(Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.use_residual = use_residual

        # define network layer modules
        if input_layer == "linear":
            self.embed = torch.nn.Linear(idim, econv_chans)
        elif input_layer == "embed":
            self.embed = torch.nn.Embedding(idim, embed_dim, padding_idx=padding_idx)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        if econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in range(econv_layers):
                ichans = (
                    embed_dim if layer == 0 and input_layer == "embed" else econv_chans
                )
                if use_batch_norm:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.BatchNorm1d(econv_chans),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
                else:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
        else:
            self.convs = None
        if elayers > 0:
            iunits = econv_chans if econv_layers != 0 else embed_dim
            self.blstm = torch.nn.LSTM(
                iunits, eunits // 2, elayers, batch_first=True, bidirectional=True
            )
        else:
            self.blstm = None

        # initialize
        self.apply(encoder_init)

        self.blstm.flatten_parameters()

    def forward(self, xs, ilens : Optional [ torch.Tensor ]=None):
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
        xs = self.embed(xs).transpose(1, 2)
        if self.convs is not None:
            for layer in self.convs:
                if self.use_residual:
                    xs += layer(xs)
                else:
                    xs = layer(xs)
        if self.blstm is None:
            return xs.transpose(1, 2)
        
        if self.training:
          ilens = torch.jit._unwrap_optional(xs)
          xp = pack_padded_sequence(xs.transpose(1, 2), ilens.cpu(), batch_first=True)
          xp, _ = self.blstm(xp)  # (B, Tmax, C)
          xs, hlens = pad_packed_sequence(xp, batch_first=True)
        else:
          xs, _ = self.blstm(xs.transpose(1,2))  # (B, Tmax, C)
          hlens = torch.tensor([xs.size(1)])

        return xs, hlens

    def inference(self, x):
        """Inference.

        Args:
            x (Tensor): The sequeunce of character ids (T,)
                    or acoustic feature (T, idim * encoder_reduction_factor).

        Returns:
            Tensor: The sequences of encoder states(T, eunits).

        """
        
        if len(x.size()) == 1:
          xs = x.unsqueeze(0)
        else:
          xs = x
        ilens = torch.tensor([xs.size(1)])

        return self.forward(xs, ilens)[0][0]
