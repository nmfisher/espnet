# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron 2 related modules for ESPnet2."""

import logging

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.hqss.loss import HQSSLoss
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.hqss.decoder import Decoder
from espnet.nets.pytorch_backend.hqss.encoder import Encoder

from espnet2.torch_utils.device_funcs import force_gatherable

from espnet2.tts.abs_tts import AbsTTS


class HQSS(AbsTTS):
    """HQSS  module for end-to-end text-to-speech."""
    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        max_oseq_len: int,
        batch_size: int,
        embed_dim: int = 512,
        econv_layers: int = 3,
        econv_chans: int = 512,
        econv_filts: int = 5,
        adim: int = 512,
        aconv_chans: int = 32,
        aconv_filts: int = 15,
        dlayers: int = 2,
        dunits: int = 1024,
        prenet_layers: int = 2,
        prenet_units: int = 256,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        output_activation: str = None,
        use_batch_norm: bool = True,
        use_residual: bool = False,
        # training related
        dropout_rate: float = 0.5,
        use_masking: bool = True,
        use_weighted_masking: bool = False,
        bce_pos_weight: float = 5.0,
        loss_type: str = "L1+L2",
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

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        
        self.loss_type = loss_type

        # define activation function for the final output
        if output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, output_activation):
            self.output_activation_fn = getattr(F, output_activation)
        else:
            raise ValueError(
                f"there is no such an activation function. " f"({output_activation})"
            )

        # set padding idx
        padding_idx = 0
        self.padding_idx = padding_idx

        # define network modules
        self.enc = Encoder(
            idim=idim,
            embed_dim=embed_dim,
            econv_chans=econv_chans,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            padding_idx=padding_idx,
        )

        print(f"econv {econv_chans} odom {odim}")

        self.dec = Decoder(
            econv_chans,
            odim,
            adim,
            max_oseq_len=max_oseq_len,
            batch_size=batch_size,
            prenet_layers=prenet_layers,
            prenet_units=prenet_units,
            postnet_layers=postnet_layers,
            postnet_chans=postnet_chans,
            postnet_filts=postnet_filts,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )
        self.hqss_loss = HQSSLoss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight,
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.

        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel

        batch_size = text.size(0)

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys = feats
        olens = feats_lengths

        # make labels for stop prediction
        labels = make_pad_mask(olens - 1).to(ys.device, ys.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # forward pass
        
        hs, hlens = self.enc(xs, ilens)
        #print(f"xs {xs.size()} ilens {ilens.size()} hs {hs.size()}")
        after_outs, before_outs, logits = self.dec(hs, ys)
        #print(f"after_outs {after_outs.size()} before_outs {before_outs.size()} logits {logits.size()} ys {ys.size()} labels {labels.size()}")
        # calculate loss (for HQSS we have copied all potential loss functions but we only use L1)
        l1_loss, mse_loss, bce_loss = self.hqss_loss(
            after_outs, before_outs, logits, ys, labels, olens
        )
        if self.loss_type == "L1+L2":
            loss = l1_loss + mse_loss + bce_loss
        elif self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = mse_loss + bce_loss
        else:
            raise ValueError(f"unknown --loss-type {self.loss_type}")

        stats = dict(
            l1_loss=l1_loss.item(),
            mse_loss=mse_loss.item(),
            bce_loss=bce_loss.item(),
        )


        
        stats.update(loss=loss.item())
        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device
        )
        return loss, stats, weight

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
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
            spembs (Optional[Tensor]): Speaker embedding (spk_embed_dim,).
            sids (Optional[Tensor]): Speaker ID (1,).
            lids (Optional[Tensor]): Language ID (1,).
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
        x = text
        y = feats
        spemb = spembs

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # inference with teacher forcing
        if use_teacher_forcing:
            assert feats is not None, "feats must be provided with teacher forcing."

            xs, ys = x.unsqueeze(0), y.unsqueeze(0)
            spembs = None if spemb is None else spemb.unsqueeze(0)
            ilens = x.new_tensor([xs.size(1)]).long()
            olens = y.new_tensor([ys.size(1)]).long()
            outs, _, _, att_ws = self._forward(
                xs=xs,
                ilens=ilens,
                ys=ys,
                olens=olens,
                spembs=spembs,
                sids=sids,
                lids=lids,
            )

            return dict(feat_gen=outs[0], att_w=att_ws[0])

        # inference
        h = self.enc.inference(x)
        if self.spks is not None:
            sid_emb = self.sid_emb(sids.view(-1))
            h = h + sid_emb
        if self.langs is not None:
            lid_emb = self.lid_emb(lids.view(-1))
            h = h + lid_emb
        if self.spk_embed_dim is not None:
            hs, spembs = h.unsqueeze(0), spemb.unsqueeze(0)
            h = self._integrate_with_spk_embed(hs, spembs)[0]
        out, prob, att_w = self.dec.inference(
            h,
            threshold=threshold,
            minlenratio=minlenratio,
            maxlenratio=maxlenratio,
            use_att_constraint=use_att_constraint,
            backward_window=backward_window,
            forward_window=forward_window,
        )

        return dict(feat_gen=out, prob=prob, att_w=att_w)

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, eunits).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, eunits) if
                integration_type is "add" else (B, Tmax, eunits + spk_embed_dim).

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        else:
            raise NotImplementedError("support only add or concat.")

        return hs
