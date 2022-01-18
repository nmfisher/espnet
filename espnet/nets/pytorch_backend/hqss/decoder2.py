#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HQSS decoder related modules."""

from unicodedata import bidirectional
import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.hqss.mol_attn2 import MOLAttn
from espnet.nets.pytorch_backend.hqss.zoneout import ZoneOutCell
from espnet.nets.pytorch_backend.hqss.prenet import Prenet
from espnet.nets.pytorch_backend.hqss.postnet import Postnet


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
        enc_odim=128,
        dec_odim=80,
        rnn_units=1024,
        prenet_layers=2,
        prenet_units=512,
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

        self.dropout_rate = dropout_rate

        self.attn_rnn_hdim = 256
        self.residual_rnn_hdim = 256
        self.attn_rnn_idim = prenet_units + self.enc_odim
        
        self.acoustic_attention_rnn = torch.nn.GRU(
            self.attn_rnn_idim, self.attn_rnn_hdim, batch_first=True, bidirectional=True)
        self.acoustic_attn = MOLAttn(self.attn_rnn_hdim * 2, num_dists=5)

        self.prosody_attention_rnn = torch.nn.GRU(
            self.attn_rnn_idim, self.attn_rnn_hdim, batch_first=True, bidirectional=True)
        self.prosody_attn = MOLAttn(self.attn_rnn_hdim * 2, num_dists=5)

        print(f"hdim {self.attn_rnn_hdim} encodim {self.enc_odim}")

        self.residual_decoders = torch.nn.LSTM(
            2*((2*self.attn_rnn_hdim) + self.enc_odim), self.residual_rnn_hdim, num_layers=1, batch_first=True, bidirectional=True)

        self.zoneout = ZoneOutCell(self.residual_decoders, zoneout_rate)

        self.pre_in = Prenet(self.dec_odim, n_units=prenet_units, dropout_rate=dropout_rate)

        self.proj_out = torch.nn.Sequential(
            torch.nn.Linear(self.residual_rnn_hdim * 2, self.dec_odim),
        )

        # FC for stop-token logits
        self.stop_prob = torch.nn.Sequential(torch.nn.Linear(self.dec_odim, 1))

        # initialize
        self.apply(decoder_init)

    def forward(self, enc_z, enc_p, ys):
        """Calculate forward propagation.

        Args:
            enc_z (Tensor): Batch of the sequences of acoustic feature encoder output (B, Tmax, enc_odim).
            enc_p (Tensor): Batch of the sequences of prosody feature encoder output (B, Tmax, enc_odim).
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

        outs = []

        logits_all = []

        # for every decoder step
        weights = []

        acoustic_context = torch.zeros(batch_size, 1, enc_z.size()[2]).to(device)
        prosody_context = torch.zeros(batch_size, 1, enc_p.size()[2]).to(device)

        for i in range(decoder_steps):
            last_out = self.pre_in(ys[:,i-1,:].unsqueeze(1) if i > 0 else torch.zeros(batch_size, 1, self.dec_odim).to(device))
            
            # acoustic features
            acoustic_attn_in = torch.cat([last_out, acoustic_context], 2) if i > 0 else torch.zeros(
                batch_size, 1, self.attn_rnn_idim)
            
            acoustic_attn_in = acoustic_attn_in.to(device)
            
            acoustic_attn_rnn_out, acoustic_attn_rnn_state = self.acoustic_attention_rnn(
                acoustic_attn_in, acoustic_attn_rnn_state if i > 0 else None)
            
            acoustic_state_unbound = torch.unbind(acoustic_attn_rnn_state,0)
            
            acoustic_state_unbound = torch.cat(acoustic_state_unbound, 1).unsqueeze(1)
            
            acoustic_context, acoustic_attn_probs = self.acoustic_attn(acoustic_state_unbound, enc_p, device=device)

            acoustic_residual_in = torch.cat([acoustic_context, acoustic_attn_rnn_out], 2).to(device)

            # prosody features
            prosody_attn_in = torch.cat([last_out, prosody_context], 2) if i > 0 else torch.zeros(
                batch_size, 1, self.attn_rnn_idim)
            
            prosody_attn_in = prosody_attn_in.to(device)
            
            prosody_attn_rnn_out, prosody_attn_rnn_state = self.prosody_attention_rnn(
                prosody_attn_in, prosody_attn_rnn_state if i > 0 else None)
            
            prosody_state_unbound = torch.unbind(prosody_attn_rnn_state,0)
            
            prosody_state_unbound = torch.cat(prosody_state_unbound, 1).unsqueeze(1)
            
            prosody_context, prosody_attn_probs = self.prosody_attn(prosody_state_unbound, enc_z, device=device)

            prosody_residual_in = torch.cat([prosody_context, prosody_attn_rnn_out], 2).to(device)

            # residual decoder
            residual_in = torch.cat([acoustic_residual_in, prosody_residual_in], 2)

            residual_out, (residual_h, residual_c) = self.zoneout(
                residual_in, (residual_h, residual_c) if i > 0 else None)

            outs += [self.proj_out(residual_out)]
            logits_all += [self.stop_prob(outs[-1])]

        outs = torch.cat(outs, 1).to(device)

        logits_all = torch.cat(logits_all, dim=1)

        return outs, torch.squeeze(logits_all, 2).to(device), weights

    def inference(
        self,
        enc_z,
        enc_p,
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
        assert len(enc_z.size()) == 2
        device = enc_z.device
        enc_z = enc_z.unsqueeze(0)
        ilens = [enc_z.size(0)]
        maxlen = int(enc_z.size(0) * maxlenratio)
        minlen = int(enc_z.size(0) * minlenratio)
        batch_size = 1

        # loop for an output sequence
        outs = []

        device = enc_z.device


        
        acoustic_context = torch.zeros(batch_size, 1, enc_z.size()[2]).to(device)
        prosody_context = torch.zeros(batch_size, 1, enc_p.size()[2]).to(device)
        i = 0
        while True:
            last_out = self.pre_in(outs[-1] if i > 0 else torch.zeros(batch_size, 1, self.dec_odim).to(device))
            
            # acoustic features
            acoustic_attn_in = torch.cat([last_out, acoustic_context], 2) if i > 0 else torch.zeros(
                batch_size, 1, self.attn_rnn_idim)
            
            acoustic_attn_in = acoustic_attn_in.to(device)
            
            acoustic_attn_rnn_out, acoustic_attn_rnn_state = self.acoustic_attention_rnn(
                acoustic_attn_in, acoustic_attn_rnn_state if i > 0 else None)
            
            acoustic_state_unbound = torch.unbind(acoustic_attn_rnn_state,0)
            
            acoustic_state_unbound = torch.cat(acoustic_state_unbound, 1).unsqueeze(1)
            
            acoustic_context, acoustic_attn_probs = self.acoustic_attn(acoustic_state_unbound, enc_p, device=device)

            acoustic_residual_in = torch.cat([acoustic_context, acoustic_attn_rnn_out], 2).to(device)

            # prosody features
            prosody_attn_in = torch.cat([last_out, prosody_context], 2) if i > 0 else torch.zeros(
                batch_size, 1, self.attn_rnn_idim)
            
            prosody_attn_in = prosody_attn_in.to(device)
            
            prosody_attn_rnn_out, prosody_attn_rnn_state = self.prosody_attention_rnn(
                prosody_attn_in, prosody_attn_rnn_state if i > 0 else None)
            
            prosody_state_unbound = torch.unbind(prosody_attn_rnn_state,0)
            
            prosody_state_unbound = torch.cat(prosody_state_unbound, 1).unsqueeze(1)
            
            prosody_context, prosody_attn_probs = self.prosody_attn(prosody_state_unbound, enc_z, device=device)

            prosody_residual_in = torch.cat([prosody_context, prosody_attn_rnn_out], 2).to(device)

            # residual decoder
            residual_in = torch.cat([acoustic_residual_in, prosody_residual_in], 2)

            residual_out, (residual_h, residual_c) = self.zoneout(
                residual_in, (residual_h, residual_c) if i > 0 else None)

            outs += [self.proj_out(residual_out)]
        
            stop_prob = self.stop_prob(outs[-1])
            # check whether to finish generation
            if int(stop_prob) > 0 or i >= 1000:

                # check mininum length
                if i < minlen:
                    continue
                if int(stop_prob) > 0:
                    print("Stopped due to stop prob")
                else:
                    print("Stopped due to max len")
                outs = torch.cat(outs, 1)
                #post_out = self.post(outs.transpose(1,2))

                after_outs = outs  # + post_out.transpose(1,2)
                return torch.squeeze(after_outs), None
            i += 1


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
