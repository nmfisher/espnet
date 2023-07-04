# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech2 related loss module for ESPnet2."""

from typing import Tuple

import torch

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.fastspeech.duration_predictor import (
    DurationPredictorLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.tacotron2.utils import make_non_pad_mask

class WLSCLoss(torch.nn.Module):
    """Loss function module for WLSC."""

    def __init__(self, use_masking:bool=True, use_weighted_masking: bool = False):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss
                calculation.

        """
        assert check_argument_types()
        super().__init__()
        
        self.use_masking = True
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" #if self.use_weighted_masking else "none"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.prior_l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.huber_criterion = torch.nn.HuberLoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(
        self,
        after_outs: torch.Tensor,
        before_outs: torch.Tensor,
        lyra_outs: torch.Tensor,
        lyra_ys: torch.Tensor,
        d_outs: torch.Tensor,
        ys: torch.Tensor,
        ds: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
        lyra_olens: torch.Tensor,
        # prior_out: torch.Tensor,
        # word_style_enc: torch.Tensor,
        sids: torch.Tensor,
        # spk_class: torch.Tensor,
        #spk_emb_preds: torch.Tensor,
        #spk_embs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor ]:
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, T_feats, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, T_feats, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ys (Tensor): Batch of target features (B, T_feats, odim).
            ds (LongTensor): Batch of durations (B, T_text).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        after_loss :  Optional[torch.Tensor] = None
        # ys = ys.view(ys.size(0), -1, 8200)
        
        if ys.dtype == torch.float:
            if self.use_masking:
                # apply mask to remove padded part
                out_masks = make_non_pad_mask(olens).to(ys.device).unsqueeze(-1)
                lyra_out_masks = make_non_pad_mask(lyra_olens).to(ys.device).unsqueeze(-1)
                masked_before_outs = before_outs.masked_select(out_masks)
                masked_lyra_outs = lyra_outs.masked_select(lyra_out_masks)
                if after_outs is not None:
                    after_outs = after_outs.masked_select(out_masks)
                # masked_ys = ys[:,:,8:].masked_select(out_masks.squeeze(-1))
                masked_ys = ys.masked_select(out_masks)
                masked_lyra_ys = lyra_ys.masked_select(lyra_out_masks)
                duration_masks = make_non_pad_mask(ilens).to(ys.device)
                d_outs = d_outs.masked_select(duration_masks)
                ds = ds.masked_select(duration_masks)    
            
            before_loss = self.huber_criterion(masked_before_outs, masked_ys).sum()
            lyra_loss = self.mse_criterion(masked_lyra_outs, masked_lyra_ys).sum()
            # xe_loss = torch.nn.functional.cross_entropy(before_outs.permute(0,3,1,2), ys[:,:,:8].long(), reduction='none')
            # xe_loss[~out_masks.squeeze(-1).squeeze(-1)] = 0
            # before_loss += xe_loss.sum()
            if after_outs is not None:
                after_loss = self.huber_criterion(after_outs, ys)
            
        elif ys.dtype ==  torch.int64 or ys.dtype == torch.int32:

            before_loss = torch.nn.functional.cross_entropy(before_outs.permute(0,3,1,2), ys, reduction='none')
            if after_outs is not None:
                after_loss = torch.nn.functional.cross_entropy(after_outs.permute(0,3,1,2), ys,reduction='none')

            if self.use_masking:
                out_masks = make_non_pad_mask(olens).to(ys.device)
                before_loss[~out_masks] = 0
                if after_outs is not None:
                    after_loss[~out_masks] = 0
        else:
            raise Exception("Unknown target dtype : " + str(ys.dtype))
        duration_loss = self.duration_criterion(d_outs, ds).mean()

        # make weighted mask and apply it
        # if self.use_weighted_masking:
            # out_weights = torch.pow(0.5, torch.arange(ys.size(-1))).to(ys.device)
            # before_loss = before_loss.mul(out_weights)
        return before_loss, after_loss if after_loss is not None else None, lyra_loss, duration_loss
