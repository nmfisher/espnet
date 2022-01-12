#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HQSS decoder related modules."""

import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.hqss.mol_attn2 import MOLAttn
from espnet.nets.pytorch_backend.hqss.zoneout import ZoneOutCell


def decoder_init(m):
    """Initialize decoder parameters."""
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("tanh"))


class Decoder(torch.nn.Module):
    """Decoder module of Spectrogram prediction network.
    This is a module of decoder of Spectrogram prediction network in HQSS
    """

    def __init__(
        self,
        enc_odim,
        dec_odim,
        adim,
        rnn_units=1024,
        prenet_layers=2,
        prenet_units=256,
        postnet_layers=5,
        postnet_chans=512,
        postnet_filts=5,
        use_batch_norm=True,
        dropout_rate=0.5,
        zoneout_rate=0.1
    ):
        """Initialize HQSS decoder module.

        Args:
            enc_odim (int): Dimension of the inputs.
            dec_odim (int): Dimension of the outputs.
            att (torch.nn.Module): Instance of attention class.
            rnn_units (int, optional): The number of decoder RNN units.
            prenet_layers (int, optional): The number of prenet layers.
            prenet_units (int, optional): The number of prenet units.
            postnet_layers (int, optional): The number of postnet layers.
            postnet_filts (int, optional): The number of postnet filter size.
            postnet_chans (int, optional): The number of postnet filter channels.
            output_activation_fn (torch.nn.Module, optional):
                Activation function for outputs.
            use_batch_norm (bool, optional): Whether to use batch normalization.
            dropout_rate (float, optional): Dropout rate.
            reduction_factor (int, optional): Reduction factor.

        """
        super(Decoder, self).__init__()

        # store the hyperparameters
        self.enc_odim = enc_odim
        self.dec_odim = dec_odim
        self.adim = adim

        self.proj_in = torch.nn.Sequential(
            torch.nn.Linear(80, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
        )
        self.attention_rnn = torch.nn.GRU(768, 512, batch_first=True)
        self.attn = MOLAttn(512, 512, self.adim)

        self.residual_decoders = torch.nn.LSTM(768, 512, num_layers=1, batch_first=True)

        #self.zoneout = ZoneOutCell(self.residual_decoders, zoneout_rate)

        self.proj_out = torch.nn.Sequential(
            torch.nn.Linear(512, self.dec_odim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.dec_odim, self.dec_odim),
            #torch.nn.Linear(256, 512),
            #  torch.nn.ReLU(),
            #torch.nn.Linear(512, self.dec_odim),
        )
        #self.proj_out = torch.nn.Linear(256, self.dec_odim)

        # FC for stop-token logits
        self.stop_prob = torch.nn.Sequential(torch.nn.Linear(self.dec_odim, 1))

        # initialize
        self.apply(decoder_init)

    def forward(self, enc_z, ys):
        """Calculate forward propagation.

        Args:
            enc_z (Tensor): Batch of the sequences of encoder output (B, Tmax, enc_odim).
            ys (Tensor):
                Batch of the sequences of padded target features (B, Lmax, odim).

        Returns:
            Tensor: Batch of output tensors after postnet (B, Lmax, odim).
            Tensor: Batch of output tensors before postnet (B, Lmax, odim).
            Tensor: Batch of logits of stop prediction (B, Lmax).
            Tensor: Batch of attention weights (B, Lmax, Tmax).

        Note:
            This computation is performed in teacher-forcing manner.

        """
        device = enc_z.device
        batch_size = int(ys.size()[0])
        decoder_steps = int(ys.size()[1])
        encoder_steps = int(enc_z.size()[1])

        outs = []

        logits_all = []

        output = torch.zeros(batch_size, 1, 512).to(device)
        context = torch.zeros(batch_size, 1, 256).to(device)
        state = torch.zeros(1,batch_size, 512).to(device)

        # for every decoder step
        weights = []
        for i in range(decoder_steps):
            input = torch.cat([output, context], 2)
            
            output, state = self.attention_rnn(input, state)

            context, _, _ = self.attn(state, enc_z, device=enc_z.device)

            residual_input = torch.cat([context, state.transpose(0,1)], 2)

            residual_out, (self.residual_c, self.residual_s) = self.residual_decoders(
                residual_input, (self.residual_c, self.residual_s) if i > 0 else None)

            last_proj = self.proj_out(residual_out)
            outs += [last_proj]
            logits = self.stop_prob(last_proj)

            logits_all += [logits]
        outs = torch.cat(outs, 1).to(device)
        logits_all = torch.cat(logits_all, dim=1)

        return outs, outs, torch.squeeze(logits_all, 2).to(device), weights

    def inference(
        self,
        h,
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=500.0,
        use_att_constraint=False,
        backward_window=None,
        forward_window=None,
    ):
        """Generate the sequence of features given the sequences of characters.

        Args:
            h (Tensor): Input sequence of encoder hidden states (T, C).
            threshold (float, optional): Threshold to stop generation.
            minlenratio (float, optional): Minimum length ratio.
                If set to 1.0 and the length of input is 10,
                the minimum length of outputs will be 10 * 1 = 10.
            minlenratio (float, optional): Minimum length ratio.
                If set to 10 and the length of input is 10,
                the maximum length of outputs will be 10 * 10 = 100.
        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).

        Note:
            This computation is performed in auto-regressive manner.

        .. _`Deep Voice 3`: https://arxiv.org/abs/1710.07654

        """
        # setup
        assert len(h.size()) == 2
        device = h.device
        enc_z = h.unsqueeze(0)
        ilens = [h.size(0)]
        maxlen = int(h.size(0) * maxlenratio)
        minlen = int(h.size(0) * minlenratio)
        batch_size = 1

        # loop for an output sequence
        i = 0
        outs = []

        device = enc_z.device

        output = torch.zeros(batch_size, 1, 512).to(device)
        context = torch.zeros(batch_size, 1, 256).to(device)
        state = torch.zeros(1,batch_size, 512).to(device)

        while True:

            input = torch.cat([output, context], 2)
            
            output, state = self.attention_rnn(input, state)

            context, _, _ = self.attn(state, enc_z, device=enc_z.device)

            residual_input = torch.cat([context, state.transpose(0,1)], 2)

            residual_out, (self.residual_c, self.residual_s) = self.residual_decoders(
                residual_input, (self.residual_c, self.residual_s) if i > 0 else None)


            last_proj = self.proj_out(residual_out)
            outs += [last_proj]

            # get logits for probability of stop token
            stop_prob = self.stop_prob(last_proj)

            # update decoder step
            i += 1

            # check whether to finish generation
            if int(stop_prob) > 0 or i >= 1000:

                # check mininum length
                if i < minlen:
                    continue
                if int(stop_prob) > 0:
                    print("Stopped due to stop prob")
                else:
                    print("Stopped due to max len")

                return torch.squeeze(torch.cat(outs, 1)), None

    def calculate_all_attentions(self, enc_z, hlens, ys):
        """Calculate all of the attention weights.

        Args:
            enc_z (Tensor): Batch of the sequences of padded hidden states (B, Tmax, enc_odim).
            hlens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor):
                Batch of the sequences of padded target features (B, Lmax, odim).

        Returns:
            numpy.ndarray: Batch of attention weights (B, Lmax, Tmax).

        Note:
            This computation is performed in teacher-forcing manner.

        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1 :: self.reduction_factor]

        # initialize hidden states of decoder
        c_list = [self._zero_state(enc_z)]
        z_list = [self._zero_state(enc_z)]
        for _ in six.moves.range(1, len(self.lstm)):
            c_list += [self._zero_state(enc_z)]
            z_list += [self._zero_state(enc_z)]
        prev_out = enc_z.new_zeros(enc_z.size(0), self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        att_ws = []
        for y in ys.transpose(0, 1):
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(enc_z, hlens, z_list[0], prev_att_w, prev_out)
            else:
                att_c, att_w = self.att(enc_z, hlens, z_list[0], prev_att_w)
            att_ws += [att_w]
            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
            enc_z = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](enc_z, (z_list[0], c_list[0]))
            for i in six.moves.range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)

        return att_ws
