# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron 2 related modules for ESPnet2."""
import os
import logging
from unittest import registerResult
import numpy as np
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
import math
import torch
import torch.nn.functional as F
import time
from espnet.nets.pytorch_backend.hqss.speaker_classifier import SpeakerClassifier

from typeguard import check_argument_types
#from espnet.nets.pytorch_backend.hqss.loc_attn import AttLoc
# from espnet.nets.pytorch_backend.rnn.attentions2 import AttLoc
# from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder
# from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder
from espnet.nets.pytorch_backend.conformer.encoder import (
    Encoder as ConformerEncoder,  # noqa: H301
)

from espnet2.tts.abs_tts import AbsTTS

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

from espnet2.tts.wlsc.loss import WLSCLoss

class SpeakerPredictor(torch.nn.Module):
    """Speaker predictor module for WLSC. 
    """

    def __init__(
        self,
        idim,
        odim,
        hidden_size=512
    ):
        """Initialize SC module.

        Args:
            idim (int) Dimension of input features (usually num_mels)
            nspeakers (int) number of speakers.
            hidden_size (int, optional) Dimension of FFN hidden size
        """
        super(SpeakerPredictor, self).__init__()

        # define network layer modules
        self.gru = torch.nn.GRU(idim, odim)

    def forward(self, xs):
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
        out, _ = self.gru(xs)
        return out[:,-1,:]
    
    def backward(self,grad_output):
      return grad_output.neg()

    def inference(self, x):
        """Inference.

        Args:
            x (Tensor): The sequeunce of character ids (T,)
                    or acoustic feature (T, idim * encoder_reduction_factor).

        Returns:
            Tensor: The sequences of encoder states(T, eunits).

        """
        raise Exception("Should not be used during inference")


class PriorEncoder(torch.nn.Module):
    def __init__(self,idim, gru_units, gru_layers):
        super().__init__()
        self.gru = torch.nn.GRU(idim, gru_units, gru_layers, batch_first=True)
        self.gru_units = gru_units
        self.gru.flatten_parameters()
        self.gru_layers = gru_layers

    def forward(self, averaged_phone_embeddings, word_embeddings):
        out, _ = self.gru(torch.cat([averaged_phone_embeddings, word_embeddings],-1))
        return out
#
# for a single sample, duration, range, feats are all N
# we want to transform this into length T = sum(duration)
#
class GaussianUpsampler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r2pi = math.sqrt(2*math.pi)
    def forward(self, feats, range, durations):
        outlen = durations.sum(1).max().item() # i.e. T
        durations = durations.unsqueeze(1).expand(durations.size(0), outlen, durations.size(1))
        c = (durations/2) + durations.cumsum(-1) #.unsqueeze(1).expand(durations.size(0), outlen, durations.size(1)) # BxTxN
        t = torch.arange(outlen).unsqueeze(1).unsqueeze(0).expand(c.size()).to(feats.device) # BxTxN
        range += 1e-6
        r = range.unsqueeze(1).expand(range.size(0), outlen, range.size(1))
        
        weights = torch.exp(-0.5 * (((t-c)/r)**2)) / (r * self.r2pi)
        weights += 1e-6
        weights_denom = weights.sum(2).unsqueeze(-1).expand(weights.size())

        weights_normalized = weights / weights_denom
        
        upsampled = torch.matmul(weights_normalized, feats)
        
        return upsampled

class WordStyleEncoder(torch.nn.Module):
    def __init__(self,idim, style_token_dim, gru_layers, num_style_tokens=15, num_attn_heads=4):
        super().__init__()
        self.num_style_tokens = num_style_tokens
        self.style_tokens = torch.nn.Parameter(torch.randn(1, self.num_style_tokens,style_token_dim))
        self.gru = torch.nn.GRU(idim, style_token_dim, gru_layers, batch_first=True)
        self.gru_units = style_token_dim
        self.gru.flatten_parameters()
        self.gru_layers = gru_layers
        self.bn = torch.nn.BatchNorm1d(idim)
        # self.attn = torch.nn.MultiheadAttention(style_token_dim,num_attn_heads, kdim=style_token_dim, vdim=style_token_dim, batch_first=True)
        self.attn = BaseMultiHeadedAttention(
            num_attn_heads,
            style_token_dim,
            0,
        )

    # feats_averaged is BxLwxF
    # phone_word_mappings is BxLp
    # durations is BxLxp
    # (B=batch size, Lw = word sequence length (i.e. num words), La= acoustic length (i.e. num frames), F=feature dimension, Lp = phonetic sequence length)
    def forward(self, feats_averaged:torch.Tensor):
        # each element in outs represents an encoding of the acoustic features for a word (being the averaged acoustic features for each phone)
        outs, _ = self.gru(feats_averaged)
        st = self.style_tokens.expand(outs.size(0), -1, -1)

        # encoded = []
        # # MHA doesn't accept 3D inputs?
        # # for i in range(outs.size(1)):
        # word = outs[:,i].unsqueeze(1) # this is the i'th word for all sequences in the batch       
        # print(st.size())
        # print(word.size())
        # attn_out, attn_w = self.attn(word,st,st,None)
        # raise Exception(attn_out.size())
        # encoded += [attn_out.unsqueeze(0)]   

        attn_out = self.attn(outs, st, st, None)
        # raise Exception(attn_out.size())
        return attn_out

class WLSC(AbsTTS):
    """WLSC module for end-to-end text-to-speech."""
    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        num_speakers: Optional[int],
        spks:Optional[int],
        spk_embed_dim:int
    ):
        """Initialize WLSC module.
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1

        # unused but need to preserve this property for compliance with ESPNet API
        self.spks = spks
        self.spk_embed_dim = spk_embed_dim
        self.langs = None
                
        # set padding idx
        self.padding_idx = 0

        self.embed_dim = 128

        # define network modules
        # self.enc = Encoder(
        #     idim=idim,
        #     embed_dim=self.embed_dim,
        #     dropout_rate=0.5,
        #     padding_idx=self.padding_idx,
        # )
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=128, padding_idx=self.padding_idx
        )

        self.word_seq_enc = torch.nn.GRU(
            bidirectional=True,
            input_size=128,
            hidden_size=64,
            batch_first=True)

        self.enc = ConformerEncoder(
                idim=idim,
                attention_dim=128,
                attention_heads=4,
                linear_units=256,
                num_blocks=6,
                dropout_rate=0.0,
                positional_dropout_rate=0.0,
                attention_dropout_rate=0.0,
                input_layer=encoder_input_layer
            )
        self.word_seq_proj_dim = 128
        # self.word_seq_proj = torch.nn.Linear(256, self.word_seq_proj_dim)
        self.style_token_dim = 128
        self.word_style_encoder = WordStyleEncoder(self.odim,self.style_token_dim, 2)

        # self.speaker_classifier = SpeakerClassifier(self.style_token_dim, spks)
        self.spk_predictor = SpeakerPredictor(self.style_token_dim, self.spk_embed_dim)
        
        # self.spk_embd = torch.nn.Embedding(num_speakers, self.word_seq_proj_dim)
        # self.spk_emb_proj = torch.nn.Linear(self.spk_embed_dim, 1024)

        # self.dec = Decoder(
        #     384,
        #     odim,
        #     AttLoc(384, 1024, odim, 32, 15)
        # )
        self.dec = ConformerEncoder(
            idim=0,
            attention_dim=128,
            attention_heads=4,
            linear_units=128,
            input_layer=None,
            num_blocks=6,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
        )

        self.prior = PriorEncoder(idim=256, gru_units=self.style_token_dim,gru_layers=2)

        self.feat_out = torch.nn.Linear(128, odim)

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=128, # concatenated size
            n_layers=2,
            n_chans=256,
            kernel_size=3,
            dropout_rate=0,
        )

        self.range_predictor = RangePredictor(
            idim=128, # concatenated size
            n_layers=2,
            n_chans=256,
            kernel_size=3,
            dropout_rate=0,
        )

        # define length regulator
        self.length_regulator = GaussianUpsampler()

        self.postnet = Postnet(
                idim=20,
                odim=odim,
                n_layers=5,
                n_chans=512,
                n_filts=5,
                use_batch_norm=True,
                dropout_rate=0,
        )
        
        self.criterion = WLSCLoss(
            use_masking=True, use_weighted_masking=False
        )


        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f"total params : {pytorch_total_params}")

    # embeddings are phonetic token embeddings, BxLpxE
    # phone_word_mappings are the indices of the word that each respetive phonetic token is derived from, BxLp
    # (B = batch dim, L = sequence length, E = embedding dim)
    # we need to average the phonetic token embeddings according to the word
    # we do this by:
    # - converting the phone<->word mappings to one-hot 
    # - left-multiply this by the transposed embeddings
    # - sum divide 
    # e.g. 
    # >>> embeddings=torch.randn(2,5,15)
    # >>> phone_word_mappings = torch.tensor([
    # ...  [0,0,1,1,1,2,0,0,0,0,0,0], # note that the zero values at the end here are padding to fill out the batch, these need to be accounted for by a mask
    # ...  [0,0,0,1,1,1,2,2,2,3,3,3],
    # ...])
    # >>> phone_word_mappings.size()
    # torch.Size([2, 12])
    # >>> word_phone_mappings=torch.nn.functional.one_hot(phone_word_mappings).transpose(1,2)
    # >>> word_phone_mappings.size()
    # torch.Size([2, 5, 12])
    # >>> product = torch.matmul(embeddings.transpose(2,1), word_phone_mappings.float())
    # >>> product.size()
    # torch.Size([2, 15, 12])
    # >>> averaged = product / torch.nn.functional.one_hot(indices).sum(0)
    # >>> averaged.size()
    # torch.Size([20, 5])
    ## check the calculation was correct by manually averaging the first three token embeddings, which correspond to the first word
    ## difference should be approx. zero
    # >>> averaged[:,0] - (data[:3,:].sum(0) / 3)
    # tensor([ 5.9605e-08, -1.4901e-08,  0.0000e+00,  0.0000e+00,  1.4901e-08,
    #     0.0000e+00,  0.0000e+00,  0.0000e+00,  1.4901e-08,  0.0000e+00,
    #     0.0000e+00,  0.0000e+00, -2.3842e-07,  0.0000e+00,  0.0000e+00,
    #     0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00])
    #
    # phone_word_mappings is BxLp
    def average(self, embeddings: torch.Tensor, phone_word_mappings:torch.Tensor, mask:torch.Tensor):
        # convert phone<->word mappings to one-hot word<->phone mappings
        # i.e. a Wmax x Pmax matrix where each row represents a word, each column a phone, and a 1 to indicate that a phone belongs a word
        # Wmax denotes the maximum number of words
        # Pmax denotes the largest phonetic symbol ID in the batch
        word_phone_mappings = torch.nn.functional.one_hot(phone_word_mappings).float().transpose(1,2)
        # zero out the masked entries so we're not inadvertently mixing up value-0 padding as word-at-index-0
        mask = mask.unsqueeze(1).expand(-1, word_phone_mappings.size(1), -1)
        word_phone_mappings[mask] = 0
        product = torch.matmul(word_phone_mappings, embeddings)
        averaged = product / (word_phone_mappings.sum(2).unsqueeze(-1) + 1e-6) # just add an epsilon to avoid divide-by-zero NaNs, these will be taken care of when the mask is applied to the loss function
        return averaged
    
    
    # 
    # for every phone embedding, we need to concatenate the style embedding for the respective word it came from
    # we do this by using the torch.gather on word_embeddings, where phone_word_mappings is used as the index
    # >>> phone_word_mappings = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4],[0, 0, 1, 2, 2, 3, 3, 3,4,4,4]])
    # >>> phone_word_mappings.size()
    # torch.Size([2, 11])
    # >>> word_embs = torch.randn(2,5,20)
    # >>> result = torch.gather(word_embs, 1, phone_word_mappings.unsqueeze(-1).expand(phone_word_mappings.size(0), phone_word_mappings.size(1), word_embs.size(2)))
    # >>> result.size()
    # torch.Size([2, 11, 20])
    ## check that this is correct
    # >>> result[0,0] - word_embs[0,0]
    # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # >>> result[0,1] - word_embs[0,0]
    # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # >>> result[0,2] - word_embs[0,0]
    # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # >>> result[0,3] - word_embs[0,1]
    # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # >>> result[1,3] - word_embs[1,2]
    # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # >>> result[1,4] - word_embs[1,3]
    # tensor([-0.9987,  0.6450, -0.5346,  0.4936, -1.3452, -0.7381, -1.7536,  0.2146,
    #      1.0220,  0.3498, -1.6553, -0.5689, -1.0752, -0.8743, -3.2089, -2.7658,
    #      0.3702,  1.5566,  0.7439,  2.1474])
    def concatenate(self, word_embeddings: torch.Tensor, phone_embeddings: torch.Tensor, phone_word_mappings: torch.Tensor, word_style_embeddings: torch.Tensor):
        # word embeddings
        word_embeddings_spread = torch.gather(word_embeddings, 1, phone_word_mappings.unsqueeze(-1).expand(phone_word_mappings.size(0), phone_word_mappings.size(1), word_embeddings.size(2)))
        # repeat for the word style embeddings
        word_style_embeddings_spread = torch.gather(word_style_embeddings, 1, phone_word_mappings.unsqueeze(-1))
        word_style_embeddings_spread = word_style_embeddings_spread.expand(phone_word_mappings.size(0), phone_word_mappings.size(1), word_embeddings.size(2))
        
        # concat = torch.cat([
        #     word_embeddings_spread, 
        #     phone_embeddings,
        #     word_style_embeddings_spread
        #     ], 
        # -1)
        # return concat
        print(word_embeddings_spread.size())
        print(phone_embeddings.size())
        print(word_style_embeddings_spread.size())
        return word_embeddings_spread + phone_embeddings + word_style_embeddings_spread

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
        # sids: torch.Tensor,
        spembs: torch.Tensor
        # spkembs: torch.Tensor
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
        batch_size = text.size(0)
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        
        durations = durations[:, : durations_lengths.max() ]
        
        # Add eos at the last of sequence
        # xs = F.pad(text, [0, 1], "constant", float(self.padding_idx))
        # for i, l in enumerate(text_lengths):
        #     xs[i, l] = self.eos
        # ilens = (text_lengths + 1).type(torch.float32)
        ilens = text_lengths
        durations_lengths = ilens
        # durations = F.pad(durations, [0, 1], "constant", 0)
        # pitch = F.pad(pitch, [0, 1], "constant", 0)

        d_masks = make_pad_mask(ilens).to(text.device)
       
        # encode style
        word_style_enc = self.word_style_encoder(feats_word_avg)

        # run word style embeddings through speaker classiifer
        # recall this has a gradient reversal layer so the intention is to decouple the word style from the speaker
        # spk_class_out = self.speaker_classifier(word_style_enc)
        spk_pred = self.spk_predictor(word_style_enc)

        # speaker embeddings
        # spk_emb = self.spk_embd(sids.view(-1))

        # encode phone sequence
        phone_enc, _ = self.enc(text, d_masks.unsqueeze(1))

        # linear project
        phone_enc_proj = phone_enc # self.word_seq_proj(phone_enc)

        # add speaker embeddings
        # spk_emb_proj_out =  self.spk_emb_proj(spembs)
        spk_emb_proj_out =  spembs
        
        # phone_enc_proj += spk_emb_proj_out.expand(phone_enc_proj.size())

        # average by word boundaries
        phone_enc_averaged = self.average(phone_enc_proj, phone_word_mappings, d_masks)

        word_enc_out, _ = self.word_seq_enc(phone_enc_averaged.detach())

        prior_out = self.prior(phone_enc_averaged.detach(), word_enc_out.detach())  

        # don't bother with masking here because we can take care of this in the loss calculation
        # (the only reason we needed to do so when averaging is because we use matmul for averaging)
        concatenated = self.concatenate(word_enc_out, phone_enc_proj, phone_word_mappings, word_style_enc)

        # concatenated = torch.cat([concatenated, spk_emb_proj_out.unsqueeze(1).expand(concatenated.size(0), concatenated.size(1), spk_emb_proj_out.size(1)) ], -1)

        concatenated += spk_emb_proj_out.unsqueeze(1).expand(concatenated.size(0), concatenated.size(1), spk_emb_proj_out.size(1))

        # predict durations & ranges        
        d_outs = self.duration_predictor(concatenated, d_masks)
        r_outs = self.range_predictor(concatenated, d_masks)

        # apply Gaussian upsampling
        upsampled = self.length_regulator(concatenated, r_outs, durations)  # (B, T_feats, adim)

        is_inference = False
        # forward decoder
        if feats_lengths is not None and not is_inference:
            h_masks = self._source_mask(feats_lengths)
        else:
            h_masks = torch.ones(1, concatenated.size(1), dtype=torch.bool).to(concatenated.device)
        
        zs, _ = self.dec(upsampled, h_masks)  # (B, T_feats, adim)

        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, T_feats, odim)

        after_outs = before_outs + self.postnet(
            before_outs.transpose(1, 2)
        ).transpose(1, 2)


        l1_loss, duration_loss, prior_loss, spk_loss = self.criterion(
            after_outs=before_outs,
            before_outs=after_outs,
            word_style_enc=word_style_enc,
            prior_out=prior_out,
            d_outs=d_outs,
            # sids=sids,
            # spk_class=spk_class_out,
            spk_preds=spk_pred,
            spk_embs=spembs,
            ys=feats,
            ds=durations,
            ilens=ilens,
            olens=feats_lengths,
        )
        loss = duration_loss + l1_loss + prior_loss + spk_loss

        stats = dict(
            l1_loss=l1_loss.item(),
            duration_loss=duration_loss.item(),
            prior_loss=prior_loss.item(),
            spk_class_loss=spk_loss.item()
        )

        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device
        )

        return loss, stats, weight
    
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
        phone_word_mappings: torch.Tensor,
        spembs: torch.Tensor,
        # sids:Optional[torch.Tensor] = None,
        durations_lengths:Optional[torch.Tensor] = None,
        feats_lengths:Optional[torch.Tensor] = None,
        pitch_lengths:Optional[torch.Tensor] = None,
        feats: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
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
        
        # text = F.pad(text, [0, 1], "constant", float(self.eos))
        ilens = torch.tensor([text.size(0)])
        
        # durations = F.pad(durations, [0, 1], "constant", 0)
        # pitch = F.pad(pitch, [0, 1], "constant", 0)
        # plens = ilens

        text = text.unsqueeze(0)
        
        phone_word_mappings = phone_word_mappings.unsqueeze(0)       

        d_masks = make_pad_mask(ilens).to(text.device)

        # sids = torch.tensor([3])

        # speaker embeddings
        # spk_emb = self.spk_embd(sids.view(-1))
        
        # encode phone sequence
        phone_enc, _ = self.enc(text, ilens)


        # linear project
        phone_enc_proj = phone_enc #self.word_seq_proj(phone_enc)

        spk_emb_proj_out =  spembs #self.spk_emb_proj(spembs)
        
        # phone_enc_proj += spk_emb_proj_out.expand(phone_enc_proj.size())

        # add speaker embeddings
        # phone_enc_proj += spk_emb.unsqueeze(1)

        # average by word boundaries
        phone_enc_averaged = self.average(phone_enc_proj, phone_word_mappings, d_masks)

        word_enc_out, _ = self.word_seq_enc(phone_enc_averaged)

        prior_out = self.prior(phone_enc_averaged, word_enc_out)  

        # prior_out += (0.1 * self.word_style_encoder.style_tokens[:,3,:].unsqueeze(1).expand(prior_out.size()))

        concatenated = self.concatenate(word_enc_out, phone_enc_proj, phone_word_mappings, prior_out)

        # concatenated = torch.cat([concatenated, spk_emb_proj_out.unsqueeze(0).expand(concatenated.size(0), concatenated.size(1), spk_emb_proj_out.size(1)) ], -1)

        concatenated += spk_emb_proj_out.unsqueeze(0).unsqueeze(0).expand(concatenated.size(0), concatenated.size(1), spk_emb_proj_out.size(0))


        # forward duration predictor and variance predictors        
        d_outs = self.duration_predictor.inference(concatenated, d_masks).int()
        r_outs = self.range_predictor.inference(concatenated, d_masks)

        # apply Gaussian upsampling
        upsampled = self.length_regulator(concatenated, r_outs, d_outs) #durations.unsqueeze(0))  # (B, T_feats, adim)
        
        h_masks = torch.ones(1, upsampled.size(1), dtype=torch.bool).to(concatenated.device)
        
        zs, _ = self.dec(upsampled,h_masks)  # (B, T_feats, adim)
        
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, T_feats, odim)

        
        after_outs = before_outs + self.postnet(
            before_outs.transpose(1, 2)
        ).transpose(1, 2)
        
        return dict(feat_gen=after_outs, prob=None, att_w=None)
        
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