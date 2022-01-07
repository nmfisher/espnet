#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HQSS decoder related modules."""

import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.hqss.mol_attn import MOLAttn
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
        enc_odim,
        dec_odim,
        adim,
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
        self.attn = MOLAttn(self.enc_odim, self.dec_odim, self.attn_dim)

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
        device = enc_z.device
        batch_size = int(ys.size()[0])
        decoder_steps = int(ys.size()[1])
        
        # B x N x odim
        after_outs_all = [] #torch.zeros(batch_size, decoder_steps, self.dec_odim, device=device)
        before_outs_all = [] # torch.zeros(batch_size, decoder_steps, self.dec_odim, device=device)
        logits_all = [] #torch.zeros(ys.size()[0], decoder_steps, device=device)

        #dec_z = torch.zeros(batch_size, decoder_steps, self.dec_odim,device=device)
        rnn_c = torch.zeros(2, batch_size,  self.dec_odim, device=device)
        rnn_s = torch.zeros(2, batch_size, self.dec_odim, device=device)
        # for every decoder step
        for i in range(decoder_steps):
            # apply prenet to last decoder output
            # (or if the first decoder step, apply to a zero frame)
            if i == 0:
                prenet_input = torch.zeros(enc_z.size()[0], self.dec_odim, device=device)
            else:
                prenet_input = torch.squeeze(after_outs_all[i-1], 1)

            prenet_output = self.prenet(prenet_input)

            # pass:
            # - encoder outputs for all timesteps,
            # - pre-net output, and
            # - index of current decoder timestep
            # to MOL attention layer.
            # returns context (B x enc_odim) and state (B x att_dim)    
            context, state, attn_w = self.attn(enc_z, prenet_output, i) 

            rnn_input = torch.cat([context, state], dim=2).to(device)
            #rnn_c = context
            #rnn_s = state

            # pass through downstream decoder layers
            before_outs, (rnn_c, rnn_s) = self.rnns(rnn_input, (rnn_c, rnn_s))



            # get logits for probability of stop token
            logits = self.stop_prob(before_outs)

            before_outs = torch.transpose(before_outs, 0,1)
            before_outs = torch.transpose(before_outs, 1,2)

            # run through postnet
            #p_out = self.postnet(before_outs)
            
            # and add
            after_outs = before_outs # + p_out # (B, odim, Lmax)
            
            after_outs_all += [ after_outs.transpose(1,2).to(device) ]
            before_outs_all += [ before_outs.transpose(1,2) ]
            logits_all += [ torch.squeeze(logits, 0) ]

        # TODO - reduction factor? 
        # the decoder will produce r frames every step
        # decoder_steps = int(ys.size()[1] / self.frames_per_step)

        # acoustic inputs to the attention RNN will be the rth frames in dec_z
        # add a zero frame for the very first decoder step
        #attn_acoustic_inputs = torch.cat([torch.zeros(batch_size, dec_z.size()[2]), dec_z[:, ::decoder_steps]

        return torch.cat(after_outs_all, dim=1).to(device), torch.cat(before_outs_all, dim=1).to(device), torch.cat(logits_all, dim=1).to(device)

    def inference(
        self,
        h,
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=50.0,
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

        # loop for an output sequence
        i = 0
        outs = []
        while True:
            # initialize hidden states of decoder
            if i == 0:
                prenet_input = torch.zeros(enc_z.size()[0], self.dec_odim, device=device)
            else:
                prenet_input = torch.squeeze(outs[i-1], dim=1)

            #idx += self.reduction_factor

            prenet_output = self.prenet(prenet_input)

            # decoder calculation
            context, state, attn_w = self.attn(enc_z, prenet_output, i) 

            rnn_input = torch.cat([context, state], dim=2).to(device)

            # pass through downstream decoder layers
            before_outs, _ = self.rnns(rnn_input)

            # get logits for probability of stop token
            stop_prob = torch.sigmoid(self.stop_prob(before_outs))

            before_outs = torch.transpose(before_outs, 0,1)
            #before_outs = torch.transpose(before_outs, 1,2)

            outs += torch.unsqueeze(before_outs, dim=0)

            # update decoder step
            i += 1
            
            # check whether to finish generation
            if int(stop_prob) > 0 or i >= maxlen:
                # check mininum length
                if i < minlen:
                    continue
                
                return torch.squeeze(torch.cat(outs, 1)), attn_w
            


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
