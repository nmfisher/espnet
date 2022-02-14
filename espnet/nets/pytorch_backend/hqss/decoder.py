#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HQSS decoder related modules."""

from unicodedata import bidirectional
import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.hqss.zoneout import ZoneOutCell
from espnet.nets.pytorch_backend.hqss.prenet import Prenet
from espnet.nets.pytorch_backend.hqss.postnet import Postnet
from typing import Optional

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
        idim,
        odim,
        attn,
        attn_p,
        dlayers=2,
        dunits=512,
        prenet_units=512,
        dropout_rate=0.5,
        zoneout_rate=0.1,
        use_batch_norm=True,
        spkr_embed_dim=64,
        device="cuda"
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
        self.idim = idim 
        self.odim = odim
        self.att = attn
        self.att_p = attn_p
        self.dunits = dunits

        self.dropout_rate = dropout_rate

        self.prenet = Prenet(self.odim, n_units=prenet_units,
                             dropout_rate=dropout_rate)
        
        self.lstm = torch.nn.ModuleList()
        for layer in six.moves.range(dlayers):
            iunits = (self.idim * 2) + prenet_units + spkr_embed_dim if layer == 0 else dunits
            lstm = torch.nn.LSTMCell(iunits, dunits)
            if zoneout_rate > 0.0:
                lstm = ZoneOutCell(lstm, zoneout_rate)
            self.lstm += [lstm]
        
        self.postnet = Postnet(odim, odim, dropout_rate=dropout_rate,use_batch_norm=use_batch_norm)
        
        iunits = (self.idim * 2) + dunits

        self.feat_out = torch.nn.Linear(
            iunits,
            self.odim,
            bias=False
        )

        # FC for stop-token logits
        self.prob_out = torch.nn.Linear(iunits, 1)

        self.apply(decoder_init)

    def _zero_state(self, hs):
        # return hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return torch.zeros(hs.size(0), self.lstm[0].hidden_size)

    def forward(self, enc_z, ilens, enc_p, plens,  ys, spk_embeds):
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
        
        # ilens = list(map(int, ilens))

        c_list = [self._zero_state(enc_z).to(enc_z.device)]
        z_list = [self._zero_state(enc_z).to(enc_z.device)]
        for _ in self.lstm[1:]:
            c_list += [self._zero_state(enc_z).to(enc_z.device)]
            z_list += [self._zero_state(enc_z).to(enc_z.device)]

        prev_out = enc_z.new_zeros(enc_z.size(0), self.odim).to(enc_z.device)

        # initialize attention
        prev_att_w : Optional [ torch.Tensor ] = None
        prev_att_w2 : Optional [ torch.Tensor ] = None
        self.att.reset(enc_z, ilens)
        self.att_p.reset(enc_p, plens)

        outs = []
        logits = []
        att_ws = []
        
        for y in ys.transpose(0, 1):

            att_c, att_w = self.att(enc_z, ilens, z_list[0], prev_att_w)

            att_c2, att_w2 = self.att_p(enc_p, plens, z_list[0], prev_att_w2)

            prenet_out = self.prenet(prev_out)
            
            xs = torch.cat([
              att_c,
              att_c2, 
              prenet_out,
              spk_embeds.squeeze(1)
            ], dim=1).to(enc_z.device)
            
            
            for i, layer in enumerate(self.lstm): 
                if i == 0:
                  z_list[0], c_list[0] = layer(xs, (z_list[0], c_list[0]))
                else:
                  z_list[i], c_list[i] = layer(
                      z_list[i - 1], (z_list[i], c_list[i])
                  )

            zcs = torch.cat([z_list[-1], att_c, att_c2], dim=1)

            outs += [self.feat_out(zcs).view(enc_z.size(0), self.odim, -1)]
            logits += [self.prob_out(zcs)]
            att_ws += [att_w[0] if isinstance(att_w, tuple) else att_w ]

            prev_att_w = att_w 

            prev_out = y

        logits = torch.cat(logits, dim=1)

        before_outs = torch.cat(outs, dim=2)
        
        att_ws = torch.stack(
            att_ws, dim=1
        ).to(enc_z.device)  # (B, Lmax, Tmax)
        
        post_out = self.postnet(before_outs)

        after_outs = before_outs + post_out

        return after_outs.transpose(1, 2), before_outs.transpose(1, 2), logits, att_ws

    def inference(
        self,
        enc_z,
        ilens,
        enc_p,
        plens,
        spk_embeds,
        threshold : float = 0.5,
        minlen: int =10,
        maxlenratio: float = 50.0,
        use_att_constraint: bool =False,
        backward_window: int =-1,
        forward_window:int=-1
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
        assert len(enc_z.size()) == 3 # batch x seq_length x hdim
        assert len(spk_embeds.size()) == 2
        
        # c_list = [self._zero_state(enc_z) for _ in self.lstm]
        # z_list = [self._zero_state(enc_z) for _ in self.lstm]
        c_list = torch.zeros(enc_z.size(0), self.lstm[0].hidden_size, len(self.lstm))
        z_list = torch.zeros(enc_z.size(0), self.lstm[0].hidden_size, len(self.lstm))

        prev_out = enc_z.new_zeros(1, self.odim).to(enc_z.device)

        # initialize attention
        prev_att_w = None
        prev_att_w2 = None
        self.att.reset(enc_z, ilens)
        self.att_p.reset(enc_p, plens)

        outs : list[torch.Tensor] = []
        att_ws = []
        stop_prob = torch.zeros(0)
        
        i = 0

        maxlen = enc_z.size(1) * maxlenratio

        # print(f"Using minlen {minlen}, max len {maxlen}, stop threshold {threshold} ")

        xs = torch.zeros(0)

        stop_prob = 0

        while stop_prob < 0.5 and i < maxlen:

          prenet_out = self.prenet(prev_out)

          att_c, att_w = self.att(
              enc_z, 
              ilens, 
              z_list[:,:,0], 
              prev_att_w, 
          )

          att_c2, att_w2 = self.att_p(
              enc_p, 
              plens, 
              z_list[:,:,0], 
              prev_att_w2, 
          )

          xs = torch.cat([
            att_c, 
            att_c2, 
            prenet_out,
            spk_embeds
          ], dim=1)
          for j, layer in enumerate(self.lstm):
            if j == 0:
              z_, c_ = layer(xs, (z_list[:,:,0], c_list[:,:,0]))
            else:
              z_, c_ = layer(
                  z_list[:,:,j-1], (z_list[:,:,j], c_list[:,:,j])
            )
            z_list[:,:,j] = z_
            c_list[:,:,j] = c_


          # for j, layer in enumerate(self.lstm):
          #   if j == 0:
          #     z_list[0], c_list[0] = layer(xs, (z_list[0], c_list[0]))
          #   else:
          #     z_list[j], c_list[j] = layer(
          #         z_list[j - 1], (z_list[j], c_list[j])
          #   )
          
          zcs = torch.cat([
            z_list[:,:,-1], 
            att_c, 
            att_c2
          ], dim=1) 

          out = self.feat_out(zcs)

          out = out.reshape(enc_z.size(0), self.odim, 1)

          outs += [ out ]

          stop_prob = torch.sigmoid(self.prob_out(zcs))[0].item()
          # if isinstance(att_w, tuple):
          #   att_ws = att_ws + [att_w[0]]
          # else:
          #   att_ws = att_ws + [att_w]

          prev_out = out.squeeze(2)
            
          i = i + 1
      
        outs_tensor = torch.cat(outs, dim=2)
        
        postnet_outs = self.postnet(outs_tensor)
        postnet_outs = postnet_outs
        outs_tensor = outs_tensor + postnet_outs

        return outs_tensor, None, None#torch.cat(att_ws, 0)


    