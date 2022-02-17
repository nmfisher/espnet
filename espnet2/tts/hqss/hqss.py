# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron 2 related modules for ESPnet2."""
import os
import logging
import numpy as np
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F
import time

from typeguard import check_argument_types
#from espnet.nets.pytorch_backend.hqss.loc_attn import AttLoc
from espnet.nets.pytorch_backend.rnn.attentions2 import AttLoc
from espnet.nets.pytorch_backend.hqss.loss import HQSSLoss
from espnet.nets.pytorch_backend.nets_utils2 import make_pad_mask
from espnet.nets.pytorch_backend.hqss.decoder import Decoder
from espnet.nets.pytorch_backend.hqss.encoder import Encoder
from espnet.nets.pytorch_backend.hqss.mol_attn import MOLAttn
from espnet.nets.pytorch_backend.hqss.speaker_classifier import SpeakerClassifier
from espnet.nets.pytorch_backend.hqss.prosody_encoder import ProsodyEncoder
from espnet.nets.pytorch_backend.hqss.residual_encoder import ResidualEncoder

from espnet2.torch_utils.device_funcs import force_gatherable

from espnet2.tts.abs_tts import AbsTTS

from espnet.nets.pytorch_backend.hqss.postnet import Postnet

class HQSS(AbsTTS):
    """HQSS  module for end-to-end text-to-speech."""
    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        num_speakers: int,
        phone_embed_dim: int = 128,
        prosody_embed_dim: int = 64,
        spkr_embed_dim: int = 64,
        cbhg_layers=1,
        cbhg_conv_layers: int = 3,
        cbhg_conv_chans: int = 512,
        cbhg_conv_filts: int = 5,
        cbhg_gru_units=128,
        adim: int = 512,
        dlayers: int = 2,
        dunits: int = 512,
        prenet_layers: int = 2,
        prenet_units: int = 512,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        output_activation: str = None,
        use_batch_norm: bool = True,
        use_residual: bool = False,
        # training related
        dropout_rate: float = 0.0,
        zoneout_rate: float = 0.0,
        use_masking: bool = True,
        use_weighted_masking: bool = False,
        bce_pos_weight: float = 5.0,
        loss_type: str = "L1+L2",
        reduction_factor: int = 1,
        num_prosody_clusters=5,
        att_filts=5
        
    ):
        """Initialize HQSS module.

        Args:
            idim (int): Dimension of the inputs.
            odim: (int) Dimension of the outputs.
            embed_dim (int): Dimension of the token embedding.
            eunits (int): Number of encoder blstm units.
            econv_layers (int): Number of encoder conv layers.
            econv_filts (int): Number of encoder conv filter size.
            econv_chans (int): Number of encoder conv filter channels.
            dlayers (int): Number of decoder lstm layers.
            dunits (int): Number of decoder lstm units.
            prenet_layers (int): Number of prenet layers.
            prenet_units (int): Number of prenet units.
            postnet_layers (int): Number of postnet layers.
            postnet_filts (int): Number of postnet filter size.
            postnet_chans (int): Number of postnet filter channels.
            output_activation (str): Name of activation function for outputs.
            adim (int): Number of dimension of mlp in attention.
            aconv_chans (int): Number of attention conv filter channels.
            aconv_filts (int): Number of attention conv filter size.
            use_batch_norm (bool): Whether to use batch normalization.
            use_concate (bool): Whether to concat enc outputs w/ dec lstm outputs.
            reduction_factor (int): Reduction factor.            
            dropout_rate (float): Dropout rate.
            zoneout_rate (float): Zoneout rate.
            use_masking (bool): Whether to mask padded part in loss calculation.
            use_weighted_masking (bool): Whether to apply weighted masking in
                loss calculation.
            bce_pos_weight (float): Weight of positive sample of stop token
                (only for use_masking=True).
            loss_type (str): Loss function type ("L1", "L2", or "L1+L2").

            guided_attn_loss_sigma (float): Sigma in guided attention loss.
            guided_attn_loss_lambda (float): Lambda in guided attention loss.

        """
        assert check_argument_types()
        super().__init__()

        print(f"Using dropout {dropout_rate}, zoneout {zoneout_rate}")

        self.langs = None
        self.spk_embed_dim = spkr_embed_dim
        self.spks = num_speakers

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.adim = adim
        
        self.loss_type = loss_type
        
        # set padding idx
        self.padding_idx = 0

        # define network modules
        self.enc = Encoder(
            idim=idim,
            embed_dim=phone_embed_dim,
            prosody_embed_dim=prosody_embed_dim,
            cbhg_conv_layers=cbhg_conv_layers,
            cbhg_conv_chans=cbhg_conv_chans,
            cbhg_conv_filts=cbhg_conv_filts,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            padding_idx=self.padding_idx,
            prenet_layers=prenet_layers,
            prenet_units=prenet_units,
            cbhg_layers=cbhg_layers,
            cbhg_gru_units=cbhg_gru_units
        )

        self.spk_embd = torch.nn.Embedding(num_speakers, self.spk_embed_dim)

        # self.residual_encoder = ResidualEncoder()

        self.dec = Decoder(
            cbhg_conv_chans,
            odim,
            # torch.nn.MultiheadAttention(cbhg_conv_chans,4, kdim=dunits, vdim=dunits),
            # torch.nn.MultiheadAttention(cbhg_conv_chans,4, kdim=dunits, vdim=dunits),
            AttLoc(
              cbhg_conv_chans,
              dunits,
              adim,
              odim, 
              att_filts
            ),
            AttLoc(
              cbhg_conv_chans,
              dunits,
              adim,
              odim, 
              att_filts
            ),
            # MOLAttn(cbhg_conv_chans, dunits, adim, num_dists=5),
            # MOLAttn(cbhg_conv_chans, dunits, adim, num_dists=5),
            dunits=dunits,
            dlayers=dlayers,
            spkr_embed_dim=spkr_embed_dim,
            prenet_layers=prenet_layers,
            prenet_units=prenet_units,
            postnet_layers=postnet_layers,
            postnet_chans=postnet_chans,
            postnet_filts=postnet_filts,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            zoneout_rate=zoneout_rate
        )

        self.speaker_classifier = SpeakerClassifier(self.odim, num_speakers)
        
        self.hqss_loss = HQSSLoss(  
            use_masking=True,
            use_weighted_masking=False,
            bce_pos_weight=bce_pos_weight,
        )

        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f"total params : {pytorch_total_params}")


    
    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor, 
        feats_lengths: torch.Tensor,
        durations:torch.Tensor,
        durations_lengths:torch.Tensor,
        pitch:torch.Tensor,
        pitch_lengths:torch.Tensor,
        sids: torch.Tensor,
        
    ) -> torch.Tensor:
        """Calculate forward propagation. 
        Since torch.jit.script is used to export the model, this method handles both training and inference.

        Args:
            text (LongTensor): Batch of padded character ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim). (training only)
            feats_lengths (LongTensor): Batch of the lengths of each target (B,). (training only)
        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.

        """

        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        durations = durations[:, : durations_lengths.max() ]
        
        batch_size = text.size(0)

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", float(self.padding_idx))
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1
        ilens = ilens.type(torch.float32)
        durations_lengths = ilens
        durations = F.pad(durations, [0, 1], "constant", 0)
        pitch = F.pad(pitch, [0, 1], "constant", 0)
        
        ys = feats 
        
        olens = feats_lengths
        

        # make labels for stop prediction
        labels = make_pad_mask(olens - 1).to(ys.device, ys.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # apply residual encoder

        # get speaker embeddings
        #spkr_emb = self.spk_embd(sids)
        spkr_emb = None
        
        # encode phone sequence
        phone_enc, _ = self.enc(xs, durations,pitch, ilens)
              
        after_outs, before_outs, logits, phone_att_w = self.dec(
          phone_enc, 
          ilens, 
          ys,
          spkr_emb
        )

        #spk_class = self.speaker_classifier(after_outs)
        spk_class = None

        # calculate loss (for HQSS we have copied all potential loss functions but we only use L1)
        l1_loss, mse_loss, bce_loss, spk_loss = self.hqss_loss(
            before_outs, after_outs, ys, logits, labels, spk_class, sids
        )

        if self.loss_type == "L1+L2":
            loss = l1_loss + bce_loss + mse_loss # + spk_loss 
        elif self.loss_type == "L1":
            loss = l1_loss + bce_loss #+ spk_loss 
        elif self.loss_type == "L2":
            loss = mse_loss + bce_loss #+ spk_loss 
        else:
            raise ValueError(f"unknown --loss-type {self.loss_type}")

        stats = dict(
            l1_loss=l1_loss,
            mse_loss=mse_loss,
            bce_loss=bce_loss,
        )

        stats.update(loss=loss.item())
        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device
        )
        return loss, stats, weight

    def export(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        durations:torch.Tensor,
        durations_lengths:torch.Tensor,
        pitch:torch.Tensor,
        pitch_lengths:torch.Tensor,
        sids:torch.Tensor,
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        
        # add eos at the last of sequence
        text = F.pad(text, [0, 1], "constant", float(self.eos))

        # get speaker embeddings
        spkr_emb = self.spk_embd(sids)

        ilens = text_lengths + 1
        ilens = ilens.type(torch.float32)

        # encode phone sequence
        phone_enc, _ = self.enc(text, durations, pitch, ilens)
        
        outs, probs, phone_att_ws, pros_att_w = self.dec.inference(phone_enc, ilens, spkr_emb)
        
        return outs, phone_att_ws 

    def inference(
        self,
        text: torch.LongTensor,
        durations:torch.Tensor,
        pitch:torch.Tensor,
        sids:torch.Tensor,
        durations_lengths:Optional[torch.Tensor] = None,
        feats_lengths:Optional[torch.Tensor] = None,
        pitch_lengths:Optional[torch.Tensor] = None,
        feats: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T_text,).
            feats (Optional[Tensor]): Feature sequence to extract style (N, idim).
            threshold (float): Threshold in inference.
            minlenratio (float): Minimum length ratio in inference.
            maxlenratio (float): Maximum length ratio in inference.
            use_att_constraint (bool): Whether to apply attention constraint.
            backward_window (int): Backward window in attention constraint.
            forward_window (int): Forward window in attention constraint.
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * prob (Tensor): Output sequence of stop probabilities (T_feats,).
                * att_w (Tensor): Attention weights (T_feats, T).

        """
        # add eos at the last of sequence
        
        text = F.pad(text, [0, 1], "constant", float(self.eos))
        ilens = torch.tensor([text.size(0)])
        ilens = ilens.type(torch.float32)
        
        durations = F.pad(durations, [0, 1], "constant", 0)
        pitch = F.pad(pitch, [0, 1], "constant", 0)
        plens = ilens

        text = text.unsqueeze(0)
        pitch = pitch.unsqueeze(0)
        #for d in [0,2,4]:
        #  durations = torch.tensor([d] * (text.size(1) - 1))
          
        durations = durations.unsqueeze(0)        

        #res_emb = self.residual_encoder(ys)

        # get speaker embeddings
        spkr_emb = self.spk_embd(sids)

        # encode phone sequence
        phone_enc, _ = self.enc(text, durations, pitch, ilens)

        outs,probs, phone_att_ws = self.dec.inference(phone_enc, ilens, spkr_emb)
        
        idx = int(time.time())
        outs.cpu().numpy().T.tofile(f"/tmp/tofile_{idx}.npy")
        phone_att_ws.cpu().numpy().tofile(f"/tmp/phone_att_weights.npy")
        text.cpu().numpy().tofile(f"/tmp/phones.npy")
        
        return dict(feat_gen=outs.squeeze(0), prob=None, att_w=phone_att_ws)
        

    
    @torch.jit.ignore
    def _plot_and_save_attention(self, att_w, filename, xtokens=None, ytokens=None):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        d = os.path.dirname(filename)
        if not os.path.exists(d):
            os.makedirs(d)
        w, h = plt.figaspect(1.0 / len(att_w))
        fig = plt.Figure(figsize=(w * 2, h * 2))
        axes = fig.subplots(1, len(att_w))
        if len(att_w) == 1:
            axes = [axes]
        for ax, aw in zip(axes, att_w):
            # plt.subplot(1, len(att_w), h)
            
            # ax.imshow(aw.cpu().detach().numpy().astype(np.float32), aspect="auto")
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            # Labels for major ticks
            if xtokens is not None:
                ax.set_xticks(np.linspace(0, len(xtokens) - 1, len(xtokens)))
                ax.set_xticks(np.linspace(0, len(xtokens) - 1, 1), minor=True)
                ax.set_xticklabels(xtokens + [""], rotation=40)
            if ytokens is not None:
                ax.set_yticks(np.linspace(0, len(ytokens) - 1, len(ytokens)))
                ax.set_yticks(np.linspace(0, len(ytokens) - 1, 1), minor=True)
                ax.set_yticklabels(ytokens + [""])
        fig.tight_layout()
        fig.savefig(filename)
        plt.clf()
        return fig
    
    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return False