# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron 2 related modules for ESPnet2."""
import os
import logging
from turtle import forward
from unittest import registerResult
import numpy as np
from typing import Dict, Optional, Sequence, Tuple, List
import math
import torch
import torch.nn.functional as F

from . import WLSC

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.conformer.encoder import (
    Encoder as ConformerEncoder,  # noqa: H301
)

from espnet.nets.pytorch_backend.hqss.postnet import Postnet
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.range_predictor import RangePredictor
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.tacotron2.utils import make_non_pad_mask
from espnet.nets.pytorch_backend.tacotron2.utils import make_pad_mask

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.fastspeech2.variance_predictor import VariancePredictor
from espnet2.tts.gst.style_encoder import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention as BaseMultiHeadedAttention,  # NOQA
)


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def backward(self,grad_output):
      return grad_output.neg()

    def forward(
        self,
        *inputs
    ) -> torch.Tensor:
        return self.inner(*inputs)
    
class Discriminator(torch.nn.Module):

    def __init__(self, odim):
        super().__init__()
        # self.rnn = torch.nn.LSTM(odim, hidden_size=512, proj_size=1, bidirectional=True)
        self.attn = torch.nn.MultiheadAttention(odim,4)

        # define network layer modules
        self.ffn = torch.nn.Sequential(
          torch.nn.Linear(odim, 512),
          torch.nn.ReLU(),
          torch.nn.Linear(512, 2),
        )

    def forward(
        self,
        student_outs:torch.Tensor,
        teacher_outs:torch.Tensor,
        olens:torch.Tensor
    ) -> torch.Tensor:
        out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(teacher_outs.device)

        attn_out_s, _ = self.attn(student_outs,student_outs,student_outs)
        ff_out_s = torch.sum(self.ffn(attn_out_s),1)
        attn_out_t, _ = self.attn(teacher_outs,teacher_outs,teacher_outs)
        ff_out_t = torch.sum(self.ffn(attn_out_t),1)
        
        return ff_out_s, ff_out_t
    

# class DiscriminatorLoss(torch.nn.Module):
#     def __init__(self, use_masking: bool = True, use_weighted_masking: bool = False):
#         """Initialize feed-forward Transformer loss module.

#         Args:
#             use_masking (bool): Whether to apply masking for padded part in loss
#                 calculation.
#             use_weighted_masking (bool): Whether to weighted masking in loss
#                 calculation.

#         """
#         assert check_argument_types()
#         super().__init__()

#         assert (use_masking != use_weighted_masking) or not use_masking
#         self.use_masking = use_masking
#         self.use_weighted_masking = use_weighted_masking

#         # define criterions
#         reduction = "none" if self.use_weighted_masking else "mean"
#         self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
#         self.mse_criterion = torch.nn.MSELoss(reduction=reduction)

#     def forward(
#         self,
#         student_class:torch.Tensor,
#         teacher_class:torch.Tensor,
#         olens:torch.Tensor
#     ) -> torch.Tensor:
#         """Calculate forward propagation.

#         Args:
#             after_outs (Tensor): Batch of outputs after postnets (B, T_feats, odim).
#             before_outs (Tensor): Batch of outputs before postnets (B, T_feats, odim).
#             d_outs (LongTensor): Batch of outputs of duration predictor (B, T_text).
#             e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
#             ys (Tensor): Batch of target features (B, T_feats, odim).
#             ds (LongTensor): Batch of durations (B, T_text).
#             es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
#             ilens (LongTensor): Batch of the lengths of each input (B,).
#             olens (LongTensor): Batch of the lengths of each target (B,).

#         Returns:
#             Tensor: L1 loss value.
#             Tensor: Duration predictor loss value.
#             Tensor: Pitch predictor loss value.
#             Tensor: Energy predictor loss value.

#         """
#         # apply mask to remove padded part
#         if self.use_masking:
#             out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(teacher_outs.device)
#             student_outs = student_outs.masked_select(out_masks)
#             teacher_outs = teacher_outs.masked_select(out_masks)
#             feats = feats.masked_select(out_masks)
#             teacher_enc = teacher_enc.masked_select(out_masks)
#             student_enc = student_enc.masked_select(out_masks)
#             teacher_dec = teacher_dec.masked_select(out_masks)
#             student_dec = student_dec.masked_select(out_masks)
            
#         # calculate loss
#         feats_loss = self.l1_criterion(student_outs, feats)
#         enc_loss = self.l1_criterion(student_enc, teacher_enc)
#         dec_loss = self.l1_criterion(student_dec, teacher_dec)
        
#         return feats_loss, enc_loss, dec_loss


class WLSCStudentLoss(torch.nn.Module):
    """Loss function module for WLSC."""

    def __init__(self, use_masking: bool = True, use_weighted_masking: bool = False):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss
                calculation.

        """
        assert check_argument_types()
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)

    def forward(
        self,
        feats:torch.Tensor,
        student_outs:torch.Tensor,
        teacher_outs:torch.Tensor,
        student_enc:torch.Tensor,
        teacher_enc:torch.Tensor,
        student_dec:torch.Tensor,
        teacher_dec:torch.Tensor,
        olens:torch.Tensor
    ) -> torch.Tensor:
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
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(teacher_outs.device)
            student_outs = student_outs.masked_select(out_masks)
            teacher_outs = teacher_outs.masked_select(out_masks)
            feats = feats.masked_select(out_masks)
            teacher_enc = teacher_enc.masked_select(out_masks)
            student_enc = student_enc.masked_select(out_masks)
            teacher_dec = teacher_dec.masked_select(out_masks)
            student_dec = student_dec.masked_select(out_masks)
            
        # calculate loss
        feats_loss = self.l1_criterion(student_outs, feats)
        enc_loss = self.l1_criterion(student_enc, teacher_enc)
        dec_loss = self.l1_criterion(student_dec, teacher_dec)
        
        return feats_loss, enc_loss, dec_loss


class WLSCStudent(WLSC):
    """WLSC module for end-to-end text-to-speech."""
    def __init__(
        self,*args, **kwargs):
        """Initialize WLSC module.
        """
        assert check_argument_types()

        teacher_ckpt = kwargs.pop("teacher_ckpt")
        teacher_conf = kwargs.pop("teacher_conf")

        super().__init__(*args, **kwargs)

        from espnet2.bin.tts_inference import Text2Speech

        teacher = Text2Speech.from_pretrained(
            model_file=teacher_ckpt,
            train_config=teacher_conf,
            vocoder_tag=None,
            device="cuda",
        ).model.tts
        teacher.eval()
        sd = teacher.state_dict().copy()
        # for k,v in teacher.named_parameters():
        #     if k not in self.state_dict():
        #         print(f"Skipping key {k}")
        #         del sd[k]
        #     elif "dec.encoders" in k and "feed_forward" in k:
        #         del sd[k]
        #         # sd[k] = torch.normal(0.0, 0.1, sd[k].size())
        # self.load_state_dict(sd, strict=False)
        self.teacher = teacher

        print(list(self.named_parameters()))
        
        # self.loss = WLSCStudentLoss()

        self.discriminator = GradientReversalLayer(Discriminator(self.odim))

        self.iter = 0

        # self.upsampled_proj = torch.nn.Linear(self.decoder_adim, self.teacher.decoder_adim)
        # self.upsampled_proj_var = torch.nn.Sequential(
        #     torch.nn.Linear(self.decoder_adim, self.teacher.decoder_adim),
        #     torch.nn.ReLU())
        # self.upsampled_proj_down = torch.nn.Linear(self.teacher.decoder_adim, self.decoder_adim)
            

        # self.dec_proj = torch.nn.Linear(self.decoder_adim, self.teacher.decoder_adim)
        # self.dec_proj_var = torch.nn.Sequential(
        #     torch.nn.Linear(self.decoder_adim, self.teacher.decoder_adim),
        #     torch.nn.ReLU())
        # self.dec_proj_down = torch.nn.Linear(self.teacher.decoder_adim, self.decoder_adim)
    
    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor, 
        feats_lengths: torch.Tensor,
        feats_word_avg: torch.Tensor, 
        feats_word_avg_lengths: torch.Tensor,
        phone_word_mappings: torch.Tensor,
        phone_word_mappings_lengths: torch.Tensor,
        durations:torch.Tensor,
        durations_lengths:torch.Tensor,
        pitch:torch.Tensor,
        pitch_lengths:torch.Tensor,
        energy:torch.Tensor,
        energy_lengths:torch.Tensor,
        sids: torch.Tensor
    ) -> torch.Tensor:
        """Calculate forward propagation. 
        Since torch.jit.script is used to export the model, this method handles both training and inference.

        Args:
            text (LongTensor): Batch of padded phone ids (B, Lp).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim). (training only).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,). (training only)
            phone_word_mappings (LongTensor): Batch of length Lp where each element is the index of the word the lengths of each target (B,). (training only)
        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.

        """
        self.iter += 1
        batch_size = text.size(0)
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        
        durations = durations[:, : durations_lengths.max() ]
        
        ilens = text_lengths
        durations_lengths = ilens

        d_masks = make_pad_mask(ilens).to(text.device)

        # encode phone sequence
        phone_enc_s, _ = self.enc(text, d_masks.unsqueeze(1))
        
        # if self.iter <= 3000 or (self.iter >= 9000 and self.iter <= 12000):
        # if self.iter % 2 == 1:
        # phone_enc_s = phone_enc_s.detach()

        with torch.no_grad():
            phone_enc_t, _ = self.teacher.enc(text, d_masks.unsqueeze(1))
            # embed speaker
            spembs_s = self.spk_embed(sids)
            spembs_t = self.teacher.spk_embed(sids)
        
            # encode style
            word_style_enc_s = self.word_style_encoder(feats_word_avg)
            word_style_enc_t = self.teacher.word_style_encoder(feats_word_avg)

            spk_class = self.speaker_classifier(word_style_enc_s)

            # linear project
            phone_enc_proj_s = self.word_seq_proj(phone_enc_s)
            phone_enc_proj_t = self.teacher.word_seq_proj(phone_enc_t)

            # average by word boundaries
            phone_enc_averaged_s = self.average(phone_enc_proj_s, phone_word_mappings, d_masks)
            phone_enc_averaged_t = self.teacher.average(phone_enc_proj_t, phone_word_mappings, d_masks)

            word_enc_out_s, _ = self.word_seq_enc(phone_enc_averaged_s.detach())
            word_enc_out_t, _ = self.teacher.word_seq_enc(phone_enc_averaged_t.detach())

            prior_out = self.prior(phone_enc_averaged_s.detach(), word_enc_out_s.detach())  

            concatenated_s = self.concatenate(word_enc_out_s, phone_enc_proj_s, phone_word_mappings, word_style_enc_s,spembs_s)
            concatenated_t = self.teacher.concatenate(word_enc_out_t, phone_enc_proj_t, phone_word_mappings, word_style_enc_t,spembs_t)

            # predict durations & ranges        
            d_outs_s = self.duration_predictor(concatenated_s, d_masks)
            r_outs_s = self.range_predictor(concatenated_s, d_masks)

            d_outs_t = self.teacher.duration_predictor(concatenated_t, d_masks)
            r_outs_t = self.teacher.range_predictor(concatenated_t, d_masks)

            # apply Gaussian upsampling
            upsampled_s = self.length_regulator(concatenated_s, r_outs_s, durations)  # (B, T_feats, adim)
            upsampled_t = self.teacher.length_regulator(concatenated_t, r_outs_t, durations)  # (B, T_feats, adim)

            # forward decoder
        
            h_masks = self._source_mask(feats_lengths)
        
            zs_t, _ = self.teacher.dec(upsampled_t.detach(), h_masks)  # (B, T_feats, adim)

        # if self.iter <= 3000 or (self.iter >= 9000 and self.iter <= 12000):
        # if self.iter % 2 == 1:
            zs_s, _ = self.dec(upsampled_t.detach(), h_masks)  # (B, T_feats, adim)    
        # else:
        # elif self.iter <= 6000 or (self.iter >= 12000 and self.iter <= 15000):
            # with torch.no_grad():
                # zs_s, _ = self.dec(upsampled_t.detach(), h_masks)  # (B, T_feats, adim)
        # else:
            # zs_s, _ = self.dec(upsampled_s.detach(), h_masks)  # (B, T_feats, adim)

            before_outs_s = self.feat_out(zs_s).view(
                zs_s.size(0), -1, self.odim
            )  # (B, T_feats, odim)

            before_outs_t = self.teacher.feat_out(zs_t).view(
                zs_s.size(0), -1, self.odim
            )  # (B, T_feats, odim)

        # after_outs_t = before_outs_t + self.teacher.postnet(
        #     before_outs_t.transpose(1, 2)
        # ).transpose(1, 2)

        # student_class, teacher_class = self.discriminator(before_outs_s, after_outs_t.detach(), feats_lengths)

        # student_loss = torch.nn.functional.cross_entropy(student_class, torch.tensor([[1.0,0.0]]).to(after_outs_t.device).expand(student_class.size()))
        # teacher_loss = torch.nn.functional.cross_entropy(teacher_class, torch.tensor([[0.0,1.0]]).to(after_outs_t.device).expand(teacher_class.size()))

        out_masks = make_non_pad_mask(feats_lengths).unsqueeze(-1).to(feats.device)
        zs_s = zs_s.masked_select(out_masks)
        zs_t = zs_t.masked_select(out_masks)
        pmasks = make_non_pad_mask(text_lengths).unsqueeze(-1).to(feats.device)
        phone_enc_s = phone_enc_s.masked_select(pmasks)
        phone_enc_t = phone_enc_t.masked_select(pmasks)

        # before_outs_s = before_outs_s.masked_select(out_masks)
        # feats = feats.masked_select(out_masks)
        l1_loss, duration_loss, prior_loss, spk_loss = self.criterion(
            after_outs=before_outs_s,
            before_outs=before_outs_s,
            word_style_enc=word_style_enc_s,
            prior_out=prior_out,
            d_outs=d_outs_s,
            ys=feats,
            ds=durations,
            ilens=ilens,
            olens=feats_lengths,
            sids=sids,spk_class=spk_class,
        )
        # loss = duration_loss + l1_loss + prior_loss + spk_loss
        dec_loss = F.mse_loss(zs_s, zs_t)
        enc_loss = F.mse_loss(phone_enc_s, phone_enc_t)
        # c_loss = F.mse_loss(concatenated_s, concatenated_t)
        # if self.iter <= 3000 or (self.iter >= 9000 and self.iter <= 12000):
        # if self.iter % 2 == 1:
        # loss = dec_loss + l1_loss 
        # else:
        # elif self.iter <= 6000 or (self.iter >= 12000 and self.iter <= 15000):
        loss = l1_loss + enc_loss
        # else:
        #     loss = dec_loss + l1_loss + enc_loss

        stats = dict(
            l1_loss=l1_loss.item(),
            duration_loss=duration_loss.item(),
            prior_loss=prior_loss.item(),
            spk_loss=spk_loss.item(),
            enc_loss=enc_loss.item(),
            # c_loss=c_loss.item()
            dec_loss=dec_loss.item()
        )
        # l1_loss = torch.nn.functional.smooth_l1_loss(before_outs_s, feats)
        # loss = l1_loss
        # # feats_loss, enc_loss, dec_loss = self.loss(
        # #     student_outs=before_outs_s,
        # #     feats=feats,
        # #     teacher_outs=after_outs_t,
        # #     olens=feats_lengths,
        # #     student_enc=upsampled_s_proj,
        # #     teacher_enc=upsampled_t,
        # #     student_dec=zs_s_proj, 
        # #     teacher_dec=zs_t
        # # )

        # # loss = feats_loss + enc_loss + dec_loss 

        # stats = dict(
        #     l1_loss=l1_loss.item(),
        #     # teacher_loss=teacher_loss.item(),
        #     # student_loss=student_loss.item()
        #     # feats_loss=feats_loss.item(),
        #     # enc_loss=enc_loss.item(),
        #     # dec_loss=dec_loss.item()
        # )

        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device
        )
        # i = 0
        # loss.backward()
        # for layer in [
        #     self.spk_embed,
        #     self.word_style_encoder,
        #     self.word_seq_proj,
        #     self.word_seq_enc,
        #     self.enc,
        #     self.dec
        # ]:
        #     total_norm = 0
        #     for p in layer.parameters():
        #         if p.grad is not None:
        #             param_norm = p.grad.detach().data.norm(2)
        #             total_norm += param_norm.item() ** 2
        #             total_norm = total_norm ** 0.5
        #     print(f"{i} {total_norm}")
        #     i += 1

        return loss, stats, weight

    # def inference(
    #     self,
    #     text: torch.LongTensor,
    #     durations:Optional[torch.Tensor] = None,
    #     pitch:Optional[torch.Tensor] = None,
    #     phone_word_mappings: torch.Tensor = None,
    #     sids:torch.Tensor = None,
    #     feats_word_avg: torch.Tensor = None, 
    #     feats_word_avg_lengths: torch.Tensor = None,
    #     durations_lengths:Optional[torch.Tensor] = None,
    #     feats_lengths:Optional[torch.Tensor] = None,
    #     pitch_lengths:Optional[torch.Tensor] = None,
    #     feats: Optional[torch.Tensor] = None,
    #     energy: Optional[torch.Tensor] = None,
    #     threshold: float = 0.5,
    #     minlenratio: float = 0.0,
    #     maxlenratio: float = 10.0,
    #     use_att_constraint: bool = False,
    #     backward_window: int = 1,
    #     forward_window: int = 3,
    #     use_teacher_forcing: bool = False,
    # ) -> Dict[str, torch.Tensor]:
    #     feats, durations, _ = self._inference(
    #         text=text, 
    #         durations=durations, 
    #         pitch=pitch, 
    #         phone_word_mappings=phone_word_mappings, 
    #         sids=sids, 
    #         feats_word_avg=feats_word_avg,
    #         durations_lengths=durations_lengths, 
    #         feats_lengths=feats_lengths, 
    #         pitch_lengths=pitch_lengths,
    #         energy=energy,
    #         threshold=threshold, 
    #         minlenratio=minlenratio, 
    #         maxlenratio=maxlenratio, 
    #         use_att_constraint=use_att_constraint, 
    #         backward_window=backward_window, 
    #         forward_window=forward_window, 
    #         use_teacher_forcing=use_teacher_forcing)
    #     return dict(feat_gen=feats, durations=durations)
    
    # def _inference(
    #     self,
    #     text: torch.LongTensor,
    #     sids:torch.Tensor,
    #     phone_word_mappings: torch.Tensor,
    #     feats_word_avg: torch.Tensor, 
    #     feats_word_avg_lengths: Optional[torch.Tensor] = None,
    #     durations:Optional[torch.Tensor] = None,
    #     pitch:Optional[torch.Tensor] = None,
    #     durations_lengths:Optional[torch.Tensor] = None,
    #     feats_lengths:Optional[torch.Tensor] = None,
    #     pitch_lengths:Optional[torch.Tensor] = None,
    #     energy: Optional[torch.Tensor] = None,        
    #     threshold: float = 0.5,
    #     minlenratio: float = 0.0,
    #     maxlenratio: float = 10.0,
    #     use_att_constraint: bool = False,
    #     backward_window: int = 1,
    #     forward_window: int = 3,
    #     use_teacher_forcing: bool = False,
    # ) -> Tuple [torch.Tensor,torch.Tensor,torch.Tensor]:
    #     """Generate the sequence of features given the sequences of characters.

    #     Args:
    #         text (LongTensor): Input sequence of characters (T_text,).
    #         feats (Optional[Tensor]): Feature sequence to extract style (N, idim).
    #         threshold (float): Threshold in inference.
    #         minlenratio (float): Minimum length ratio in inference.
    #         maxlenratio (float): Maximum length ratio in inference.
    #         use_att_constraint (bool): Whether to apply attention constraint.
    #         backward_window (int): Backward window in attention constraint.
    #         forward_window (int): Forward window in attention constraint.
    #         use_teacher_forcing (bool): Whether to use teacher forcing.

    #     Returns:
    #         Dict[str, Tensor]: Output dict including the following items:
    #             * feat_gen (Tensor): Output sequence of features (T_feats, odim).
    #             * prob (Tensor): Output sequence of stop probabilities (T_feats,).
    #             * att_w (Tensor): Attention weights (T_feats, T).

    #     """
    #     # add eos at the last of sequence
    #     # text = F.pad(text, [0, 1], "constant", float(self.eos))
    #     ilens = torch.tensor([text.size(0)])
        
    #     # durations = F.pad(durations, [0, 1], "constant", 0)
    #     # pitch = F.pad(pitch, [0, 1], "constant", 0)
    #     # plens = ilens

    #     text = text.unsqueeze(0)
        
    #     phone_word_mappings = phone_word_mappings.unsqueeze(0)       

    #     # feats_word_avg = feats_word_avg.unsqueeze(0)

    #     d_masks = make_pad_mask(ilens).to(text.device)
        
    #     # encode phone sequence
    #     phone_enc, _ = self.enc(text, ilens)

    #     # linear project
    #     phone_enc_proj = self.word_seq_proj(phone_enc)

    #     spembs = self.spk_embed(sids).unsqueeze(0)

    #     # average by word boundaries
    #     phone_enc_averaged = self.average(phone_enc_proj, phone_word_mappings, d_masks)

    #     word_enc_out, _ = self.word_seq_enc(phone_enc_averaged)

    #     prior_out = self.prior(phone_enc_averaged, word_enc_out)  

    #     # word_style_enc = self.word_style_encoder(feats_word_avg)

    #     concatenated = self.concatenate(word_enc_out, phone_enc_proj, phone_word_mappings, \
    #     prior_out, 
    #     #word_style_enc,
    #     spembs)
    #     concatenated = concatenated.view(phone_enc_proj.size(0), phone_enc_proj.size(1), self.decoder_adim)
    #     # forward duration predictor and variance predictors        
    #     d_outs = self.duration_predictor.inference(concatenated, d_masks).int()
        
    #     r_outs = self.range_predictor.inference(concatenated, d_masks)

    #     # apply Gaussian upsampling
    #     #upsampled = self.length_regulator(concatenated, r_outs, d_outs) #durations.unsqueeze(0))  # (B, T_feats, adim)
    #     upsampled_s = self.length_regulator(concatenated, torch.zeros(r_outs.size()), durations.unsqueeze(0))  # (B, T_feats, adim)
    #     upsampled = upsampled_s
        
    #     h_masks = torch.ones(1, upsampled.size(1), dtype=torch.bool).to(concatenated.device)
        
    #     zs, _ = self.dec(upsampled,h_masks)  # (B, T_feats, adim)
                
    #     before_outs = self.feat_out(zs).view(
    #         zs.size(0), -1, self.odim
    #     )  # (B, T_feats, odim)

    #     after_outs = before_outs + self.postnet(
    #         before_outs.transpose(1, 2)
    #     ).transpose(1, 2)
        
    #     return before_outs, d_outs, phone_word_mappings
    
    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(ilens.device)
        return x_masks.unsqueeze(-2)

    def export(
        self,
        text: torch.Tensor,
        sids:torch.Tensor,
        phone_word_mappings:torch.Tensor,
        feats_word_avg: torch.Tensor 
    ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        return self._inference(text, sids=sids,phone_word_mappings=phone_word_mappings,feats_word_avg=feats_word_avg)     

    
    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return False
