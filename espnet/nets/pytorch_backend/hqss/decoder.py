#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HQSS decoder related modules."""

import six

import torch
import torch.nn.functional as F

from mol_attn import MOLAttn
from prenet import Prenet
from postnet import Postnet

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
        max_oseq_len,
        batch_size,
        rnn_units=1024,
        prenet_layers=2,
        prenet_units=256,
        postnet_layers=5,
        postnet_chans=512,
        postnet_filts=5,
        use_batch_norm=True,
        dropout_rate=0.5
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
        self.attn_dim = adim
        self.rnn_units = rnn_units

        # define prenet
        self.prenet = Prenet(
            dec_odim,
            n_layers=prenet_layers,
            n_units=dec_odim,
            dropout_rate=dropout_rate,
        )

        # define alignment attention RNN/attention layer
        self.attn = MOLAttn(self.enc_odim, self.dec_odim, self.attn_dim, batch_size)

        # define decoder RNNs 
        
        # check this? if we are concatenating context + state from attention, size will be B x (enc_odim + att_dim)
        self.rnns = torch.nn.LSTM(enc_odim + self.attn_dim, self.dec_odim, num_layers=2)

        # define postnet
        self.postnet = Postnet(
            enc_odim,
            self.dec_odim,
            n_layers=postnet_layers,
            n_chans=postnet_chans,
            n_filts=postnet_filts,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )
        
        # FC for stop-token logits 
        # input will be 
        self.stop_prob = torch.nn.Linear(self.dec_odim, 1)

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
        batch_size = int(ys.size()[0])
        decoder_steps = int(ys.size()[1])
        
        # B x N x odim
        after_outs_all = torch.zeros(batch_size, decoder_steps, self.dec_odim)
        before_outs_all = torch.zeros(batch_size, decoder_steps, self.dec_odim)
        logits_all = torch.zeros(ys.size()[0], decoder_steps)

        # for every decoder step
        for i in range(decoder_steps):
            # apply prenet to last decoder output
            # (or if the first decoder step, apply to a zero frame)
            if i == 0:
                dec_z = self.prenet(torch.zeros(enc_z.size()[0], self.dec_odim))
            else:
                dec_z = self.prenet(after_outs_all[:, i-1, :])
    
            # pass:
            # - encoder outputs for all timesteps,
            # - pre-net output, and
            # - index of current decoder timestep
            # to MOL attention layer.
            # returns context (B x enc_odim) and state (B x att_dim)
            context, state = self.attn(enc_z, dec_z, i) 

            #print(context.size())
            #print(state.size())

            rnn_input = torch.cat([context, state], dim=2)

            # pass through downstream decoder layers
            before_outs, _ = self.rnns(rnn_input)

            #print(before_outs.size())

            # get logits for probability of stop token
            logits = self.stop_prob(before_outs)

            before_outs = torch.transpose(before_outs, 0,1)
            before_outs = torch.transpose(before_outs, 1,2)
        
            # run through postnet
            p_out = self.postnet(before_outs)
            # and add
            after_outs = before_outs + p_out # (B, odim, Lmax)
            
            #print(after_outs.size())
            #print(before_outs.size())
            #print(logits.size())
            #print(logits_all.size())
            after_outs_all[:,i,:] = torch.squeeze(after_outs)
            before_outs_all[:,i,:] = torch.squeeze(before_outs)
            logits_all[:,i] = torch.squeeze(logits)

        # TODO - reduction factor? 
        # the decoder will produce r frames every step
        # decoder_steps = int(ys.size()[1] / self.frames_per_step)

        # acoustic inputs to the attention RNN will be the rth frames in dec_z
        # add a zero frame for the very first decoder step
        #attn_acoustic_inputs = torch.cat([torch.zeros(batch_size, dec_z.size()[2]), dec_z[:, ::decoder_steps]

        return after_outs_all, before_outs_all, logits_all

    def inference(
        self,
        h,
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
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
            use_att_constraint (bool):
                Whether to apply attention constraint introduced in `Deep Voice 3`_.
            backward_window (int): Backward window size in attention constraint.
            forward_window (int): Forward window size in attention constraint.

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
        enc_z = h.unsqueeze(0)
        ilens = [h.size(0)]
        maxlen = int(h.size(0) * maxlenratio)
        minlen = int(h.size(0) * minlenratio)

        # initialize hidden states of decoder
        c_list = [self._zero_state(enc_z)]
        z_list = [self._zero_state(enc_z)]
        for _ in six.moves.range(1, len(self.lstm)):
            c_list += [self._zero_state(enc_z)]
            z_list += [self._zero_state(enc_z)]
        prev_out = enc_z.new_zeros(1, self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # setup for attention constraint
        if use_att_constraint:
            last_attended_idx = 0
        else:
            last_attended_idx = None

        # loop for an output sequence
        idx = 0
        outs, att_ws, probs = [], [], []
        while True:
            # updated index
            idx += self.reduction_factor

            # decoder calculation
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(
                    enc_z,
                    ilens,
                    z_list[0],
                    prev_att_w,
                    prev_out,
                    last_attended_idx=last_attended_idx,
                    backward_window=backward_window,
                    forward_window=forward_window,
                )
            else:
                att_c, att_w = self.att(
                    enc_z,
                    ilens,
                    z_list[0],
                    prev_att_w,
                    last_attended_idx=last_attended_idx,
                    backward_window=backward_window,
                    forward_window=forward_window,
                )

            att_ws += [att_w]
            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
            enc_z = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](enc_z, (z_list[0], c_list[0]))
            for i in six.moves.range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            zcs = (
                torch.cat([z_list[-1], att_c], dim=1)
                if self.use_concate
                else z_list[-1]
            )
            outs += [self.feat_out(zcs).view(1, self.odim, -1)]  # [(1, odim, r), ...]
            probs += [torch.sigmoid(self.prob_out(zcs))[0]]  # [(r), ...]
            if self.output_activation_fn is not None:
                prev_out = self.output_activation_fn(outs[-1][:, :, -1])  # (1, odim)
            else:
                prev_out = outs[-1][:, :, -1]  # (1, odim)
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w
            if use_att_constraint:
                last_attended_idx = int(att_w.argmax())

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = torch.cat(outs, dim=2)  # (1, odim, L)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                att_ws = torch.cat(att_ws, dim=0)
                break

        if self.output_activation_fn is not None:
            outs = self.output_activation_fn(outs)

        return outs, probs, att_ws

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
