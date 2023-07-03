# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron 2 related modules for ESPnet2."""
import os
import logging
from turtle import forward
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
import torch.nn.utils.prune as prune

from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)


from espnet.nets.pytorch_backend.hqss.speaker_classifier import SpeakerClassifier

from typeguard import check_argument_types

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
from espnet2.tts.wlsc.grad_reversal_layer import GradientReversalLayer

class SpeakerEmbeddingPredictor(torch.nn.Module):
    """Speaker classifier module for HQSS. Simply FFN with gradient reversal.
    """

    def __init__(
        self,
        idim,
        odim
    ):
        """Initialize SC module.

        Args:
            idim (int) Dimension of input features (usually num_mels)
            nspeakers (int) number of speakers.
            hidden_size (int, optional) Dimension of FFN hidden size
        """
        super(SpeakerEmbeddingPredictor, self).__init__()
        # store the hyperparameters
        # self.cnn = torch.nn.Conv1d(idim,odim,3)       
        # self.rnn = torch.nn.GRU(idim,odim,3)       
        # self.attn = BaseMultiHeadedAttention(
        #     8,
        #     odim,
        #     0.05,
        # )
        self.enc = ConformerEncoder(
            idim=idim,
            attention_dim=256,
            attention_heads=8,
            num_blocks=5,
            dropout_rate=0.5,
            positional_dropout_rate=0.5,
            attention_dropout_rate=0.5,
            input_layer="linear",
            # positionwise_layer_type=encoder_positionwise_layer_type,
            # positionwise_conv_kernel_size=encoder_conv_kernel_size, 
            pos_enc_layer_type="rel_pos",   
            selfattention_layer_type="rel_selfattn", 
            activation_type="swish",             
            macaron_style=False,
            # use_cnn_module=encoder_use_cnn_module,
            # cnn_module_kernel=7
        )

    def forward(self, xs, mask):
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
        outs, m = self.enc(xs,mask)
        return outs.sum(1)
        # return self.cnn(xs.transpose(1,2)).sum(-1)
        # out, _ = self.rnn(xs)
        # attn_out = self.attn(out, out, out, mask)
        # return attn_out.sum(1)

    
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
        out, _ = self.gru(torch.cat([averaged_phone_embeddings, word_embeddings],2))
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
        
        durations = durations.unsqueeze(1).expand(-1, outlen, -1).float()
        
        c = (durations/2) + durations.cumsum(2) # BxTxN
        t = torch.arange(outlen,dtype=torch.float32).unsqueeze(1).unsqueeze(0).expand(c.size()).to(feats.device) # BxTxN
        range += 1e-6
        r = range.unsqueeze(1).expand(range.size(0), outlen, range.size(1))
        
        weights = torch.exp(-0.5 * (((t-c)/r)**2)) / (r * self.r2pi)
        weights += 1e-6
        weights_denom = weights.sum(2).unsqueeze(2).expand(weights.size())

        weights_normalized = weights / weights_denom
        
        upsampled = torch.matmul(weights_normalized, feats)
        
        return upsampled.view(feats.size(0), outlen, feats.size(2))

class WordStyleEncoder(torch.nn.Module):
    def __init__(self,idim, style_token_dim, gru_layers, num_style_tokens=15, num_attn_heads=4,rnn_dropout=0.1, mha_dropout=0.1):
        super().__init__()
        self.num_style_tokens = num_style_tokens
        self.style_tokens = torch.nn.Parameter(torch.randn(1, self.num_style_tokens,style_token_dim))
        self.gru = torch.nn.GRU(idim, style_token_dim, gru_layers, batch_first=True, dropout=rnn_dropout)
        self.gru_units = style_token_dim
        self.gru.flatten_parameters()
        self.gru_layers = gru_layers
        # self.attn = torch.nn.MultiheadAttention(style_token_dim,num_attn_heads, kdim=style_token_dim, vdim=style_token_dim, batch_first=True)
        self.attn = BaseMultiHeadedAttention(
            num_attn_heads,
            style_token_dim,
            mha_dropout,
        )

    # feats_averaged is BxLwxF
    # phone_word_mappings is BxLp
    # durations is BxLxp
    # (B=batch size, Lw = word sequence length (i.e. num words), La= acoustic length (i.e. num frames), F=feature dimension, Lp = phonetic sequence length)
    def forward(self, feats_averaged:torch.Tensor):
        # each element in outs represents an encoding of the acoustic features for a word (being the averaged acoustic features for each phone)
        outs, _ = self.gru(feats_averaged)
        st = self.style_tokens.expand(outs.size(0), -1, -1)
        attn_out = self.attn(outs, st, st, None)
        return attn_out

class WLSC(AbsTTS):
    """WLSC module for end-to-end text-to-speech."""
    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        num_speakers: Optional[int],
        num_prosody_clusters:Optional[int],
        spks:Optional[int],
        spk_embed_dim:Optional[int]=256,
        spk_classifier_size:Optional[int]=256,
        phone_embed_dim:Optional[int]=256,
        word_seq_embed_dim:Optional[int]=256,
        word_seq_proj_dim:Optional[int]=256,
        style_token_dim:Optional[int]=256,
        word_seq_encoder_dropout:Optional[float]=0.1,
        encoder_adim:Optional[int]=256,
        encoder_linear_units:Optional[int]=2048,
        encoder_aheads:Optional[int]=2,
        encoder_numblocks:Optional[int]=5,
        encoder_dropout_rate:Optional[float]=0.1,
        encoder_positional_dropout_rate:Optional[float]=0.1,
        encoder_attention_dropout_rate:Optional[float]=0.1,
        encoder_conv_kernel_size:Optional[int]=3,
        encoder_positionwise_conv_kernel_size=21,
        encoder_positionwise_layer_type="conv1d",
        encoder_use_cnn_module=False,
        encoder_macaron_style=False,
        encoder_cnn_module_kernel=3,
        style_encoder_aheads:Optional[int]=2,
        decoder_adim:Optional[int]=1024,
        decoder_aheads:Optional[int]=2,
        decoder_numblocks:Optional[int]=5,
        decoder_dropout_rate:Optional[float]=0.1,
        decoder_positional_dropout_rate:Optional[float]=0.1,
        decoder_attention_dropout_rate:Optional[float]=0.1,
        decoder_positionwise_layer_type="conv1d", 
        decoder_positionwise_conv_kernel_size=21,
        decoder_macaron_style=False,
        decoder_use_cnn_module=False,
        decoder_linear_units=2048,
        decoder_cnn_module_kernel=31,
        prior_layers:Optional[int]=2,
        duration_predictor_layers:Optional[int]=2,
        duration_predictor_chans:Optional[int]=256,
        duration_predictor_kernel_size:Optional[int]=3,
        duration_predictor_dropout=0.1,
        range_predictor_layers:Optional[int]=2,
        range_predictor_chans:Optional[int]=256,
        range_predictor_kernel_size:Optional[int]=3,
        range_predictor_dropout=0.1,
        postnet_layers:Optional[int]=5,
        postnet_chans:Optional[int]=512,
        postnet_filts:Optional[int]=5,
        postnet_dropout:Optional[float]=0.1,
        proj_up_down:Optional[bool]=False,
    ):
        """Initialize WLSC module.
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1

        self.spks = spks

        self.num_prosody_clusters = num_prosody_clusters
        
        # we can't set self.spk_embed_dim because ESPnet will expect speaker embeddings to be passed in
        self.spk_embed_dim = spk_embed_dim
        self.spk_embed_layer_dim = spk_embed_dim
        
        # unused but need to preserve this property for compliance with ESPNet API
        self.langs = None

        self.spk_embed = torch.nn.Embedding(
            num_embeddings=self.spks+1, embedding_dim=self.spk_embed_layer_dim, padding_idx=self.spks
        )

        #self.speaker_classifier = SpeakerClassifier(self.spk_embed_layer_dim, self.spks, spk_classifier_size)
        #self.speaker_emb_predictor = SpeakerEmbeddingPredictor(style_token_dim, self.spk_embed_dim)

        # self.gradient_reversal_layer = GradientReversalLayer(0.1)
                
        # set padding idx
        self.padding_idx = 0

        self.phone_embed_dim = phone_embed_dim
        self.pitch_embed_dim = phone_embed_dim
        self.energy_embed_dim = phone_embed_dim

        self.pitch_embed = torch.nn.Embedding(
            num_embeddings=self.num_prosody_clusters+1,
            embedding_dim=phone_embed_dim,
            padding_idx=self.num_prosody_clusters
        )

        self.energy_embed = torch.nn.Embedding(
            num_embeddings=self.num_prosody_clusters+1,
            embedding_dim=phone_embed_dim,
            padding_idx=self.num_prosody_clusters
        )


        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=self.phone_embed_dim, padding_idx=self.padding_idx
        )

        self.word_seq_embedding_dim = word_seq_embed_dim 

        # self.word_seq_enc = torch.nn.GRU(
        #     bidirectional=True,
        #     input_size=self.phone_embed_dim,
        #     hidden_size=self.word_seq_embedding_dim,
        #     batch_first=True,
        #     dropout=word_seq_encoder_dropout)

        self.enc = ConformerEncoder(
            idim=phone_embed_dim,
            attention_dim=encoder_adim,
            attention_heads=encoder_aheads,
            num_blocks=encoder_numblocks,
            dropout_rate=encoder_dropout_rate,
            positional_dropout_rate=encoder_positional_dropout_rate,
            attention_dropout_rate=encoder_attention_dropout_rate,
            input_layer=encoder_input_layer,
            linear_units=encoder_linear_units,
            positionwise_layer_type=encoder_positionwise_layer_type,
            positionwise_conv_kernel_size=encoder_positionwise_conv_kernel_size, 
            pos_enc_layer_type="rel_pos",   
            selfattention_layer_type="rel_selfattn", 
            activation_type="swish",             
            macaron_style=encoder_macaron_style,
            use_cnn_module=encoder_use_cnn_module,
            cnn_module_kernel=encoder_cnn_module_kernel
        )
        self.enc_proj = torch.nn.Linear(encoder_adim, decoder_adim)
        # self.word_seq_proj_dim = word_seq_proj_dim
        # self.word_seq_proj = torch.nn.Linear(self.word_seq_embedding_dim, self.word_seq_proj_dim)
        # self.style_token_dim = style_token_dim
        # self.word_style_encoder = WordStyleEncoder(self.odim,self.style_token_dim, style_encoder_aheads)
        self.encoder_adim = encoder_adim
        self.decoder_adim = decoder_adim

        self.dec = ConformerEncoder(
            idim=encoder_adim,
            attention_dim=decoder_adim,
            attention_heads=decoder_aheads,
            input_layer=None,
            num_blocks=decoder_numblocks,
            dropout_rate=decoder_dropout_rate,
            linear_units=decoder_linear_units,
            positional_dropout_rate=decoder_positional_dropout_rate,
            attention_dropout_rate=decoder_attention_dropout_rate,
            positionwise_layer_type=decoder_positionwise_layer_type,
            positionwise_conv_kernel_size=decoder_positionwise_conv_kernel_size, 
            pos_enc_layer_type="rel_pos",   
            selfattention_layer_type="rel_selfattn", 
            activation_type="swish",             
            macaron_style=decoder_macaron_style,
            use_cnn_module=decoder_use_cnn_module,
            cnn_module_kernel=decoder_cnn_module_kernel,
        )
        concat_dim = self.phone_embed_dim # + spk_embed_dim + self.pitch_embed_dim  + self.energy_embed_dim  #+ self.word_seq_embedding_dim +  # + self.style_token_dim
        # self.prior = PriorEncoder(idim=self.phone_embed_dim + self.word_seq_embedding_dim * 2, gru_units=self.style_token_dim,gru_layers=prior_layers)

        self.feat_out = torch.nn.Linear(decoder_adim, self.odim)

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=concat_dim, # concatenated size # TODO fix this
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout,
        )

        self.range_predictor = RangePredictor(
            idim=concat_dim, # concatenated size # TODO fix this
            n_layers=range_predictor_layers,
            n_chans=range_predictor_chans,
            kernel_size=range_predictor_kernel_size,
            dropout_rate=range_predictor_dropout,
        )

        # define length regulator
        self.length_regulator = GaussianUpsampler()

        self.postnet = Postnet(
                idim=odim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=True,
                dropout_rate=postnet_dropout,
        )
        
        self.criterion = WLSCLoss(
            use_masking=True, use_weighted_masking=False
        )

        for c in [ 
            # self.spk_embed, 
            #self.speaker_emb_predictor,
            self.duration_predictor, 
            self.range_predictor, 
            #self.prior, 
            self.feat_out, 
            self.enc, 
            self.dec, 
            #self.word_style_encoder, 
            # self.word_seq_enc, 
            # self.word_seq_proj 
            ]:
            pytorch_total_params = sum(p.numel() for p in c.parameters())
            print(f"params for {type(c).__name__} : {pytorch_total_params}")
        # self.proj_up_down = proj_up_down
        
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
        word_phone_mappings = torch.nn.functional.one_hot(phone_word_mappings,num_classes=phone_word_mappings.max()+1).float().transpose(1,2)
        
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
    def concatenate(self, word_embeddings: torch.Tensor, phone_embeddings: torch.Tensor, phone_word_mappings: torch.Tensor, word_style_embeddings: torch.Tensor, speaker_embeddings:torch.Tensor):
        phone_word_mappings = phone_word_mappings.unsqueeze(2)
        # word embeddings
        word_embeddings_indices = phone_word_mappings.expand(phone_word_mappings.size(0), phone_word_mappings.size(1), self.word_seq_embedding_dim)
        
        if self.training:
            word_embeddings_spread = torch.gather(word_embeddings, 1, word_embeddings_indices)    
        else:
            # to preserve shape inferencing (which is needed due to transpose calls later down the line) in ONNX export, we need to reshape
            word_embeddings_spread = torch.gather(word_embeddings, 1, word_embeddings_indices).reshape(phone_embeddings.size(0),phone_embeddings.size(1),self.word_seq_embedding_dim)
        
        # repeat for the word style embeddings
        word_style_embeddings_indices = phone_word_mappings.expand(phone_word_mappings.size(0), phone_word_mappings.size(1), self.style_token_dim)
        
        if self.training:
            word_style_embeddings_spread = torch.gather(word_style_embeddings, 1, word_style_embeddings_indices)    
        else:
            word_style_embeddings_spread = torch.gather(word_style_embeddings, 1, word_style_embeddings_indices).reshape(phone_embeddings.size(0),phone_embeddings.size(1),self.style_token_dim)

        concat = torch.cat([
             word_embeddings_spread, 
             phone_embeddings,
             word_style_embeddings_spread,
             speaker_embeddings.expand(-1,phone_embeddings.size(1), -1)
             ], 2)
        return concat
#        return word_embeddings_spread + phone_embeddings + word_style_embeddings_spread + speaker_embeddings.expand(-1,phone_embeddings.size(1), -1)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor, 
        feats_lengths: torch.Tensor,        
        durations: torch.Tensor,
        durations_lengths:torch.Tensor,
        sids: torch.Tensor,
        spembs: torch.Tensor,
        feats_word_avg: Optional[torch.Tensor]=None, 
        feats_word_avg_lengths: Optional[torch.Tensor]=None,
        phone_word_mappings: Optional[torch.Tensor]=None,
        phone_word_mappings_lengths: Optional[torch.Tensor]=None,
        pitch:Optional[torch.Tensor]=None,
        pitch_lengths:Optional[torch.Tensor]=None,
        energy:Optional[torch.Tensor]=None,
        energy_lengths:Optional[torch.Tensor]=None,
        # is_inference: bool,
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
        feats = feats[:, : feats_lengths.max()].long()  # for data-parallel
        
        durations = durations[:, : durations_lengths.max() ]

        # Add eos at the last of sequence
        # xs = F.pad(text, [0, 1], "constant", float(self.padding_idx))
        # for i, l in enumerate(text_lengths):
        #     xs[i, l] = self.eos
        # ilens = (text_lengths + 1).type(torch.float32)
        ilens = text_lengths

        # durations = F.pad(durations, [0, 1], "constant", 0)
        # pitch = F.pad(pitch, [0, 1], "constant", 0)

        d_masks = make_pad_mask(ilens).to(text.device)

        # embed speaker
        spembs = self.spk_embed(sids)

        #pitch_embs = self.pitch_embed(pitch)
        #energy_embs = self.energy_embed(energy)
       
        # encode style
        # word_style_enc = self.word_style_encoder(feats_word_avg)

        # word_masks = make_pad_mask(feats_word_avg_lengths).to(text.device)

        # spk_class = self.speaker_classifier(word_style_enc)
        # spk_emb_preds = self.gradient_reversal_layer(self.speaker_emb_predictor(word_style_enc, word_masks.unsqueeze(2)))
        # don't allow gradient flow when using shuffled speaker embeddings, we just want to see how predictive the style embeddings are of the speaker identity
        # spk_emb_preds = self.speaker_emb_predictor(word_style_enc.detach(), word_masks.unsqueeze(2))

        # encode phone sequence
        phone_enc, _ = self.enc(text, d_masks.unsqueeze(1))

        # linear project
        # phone_enc_proj = self.word_seq_proj(phone_enc)

        # average by word boundaries
        # phone_enc_averaged = self.average(phone_enc_proj, phone_word_mappings, d_masks)

        # word_enc_out, _ = self.word_seq_enc(phone_enc_averaged.detach())
        
        # prior_out = self.prior(phone_enc_averaged.detach(), word_enc_out.detach())  

        concatenated = phone_enc #torch.cat([
            #phone_enc, 
         #   spembs.expand(-1,phone_enc.size(1), -1), 
         #   pitch_embs, 
         #   energy_embs,
            #], 
        #2)

        # don't bother with masking here because we can take care of this in the loss calculation
        # (the only reason we needed to do so when averaging is because we use matmul for averaging)
        # concatenated = self.concatenate(
        #     word_enc_out, 
        #     phone_enc_proj, 
        #     phone_word_mappings, 
        #     # word_style_enc,
        #     spembs
        # )

        # predict durations & ranges        
        d_outs = self.duration_predictor(concatenated, d_masks)
        r_outs = self.range_predictor(concatenated, d_masks)
        
        # apply Gaussian upsampling
        upsampled = self.length_regulator(concatenated, r_outs, durations)  # (B, T_feats, adim)
        
        h_masks = self._source_mask(feats_lengths)        
        
        zs, _ = self.dec(self.enc_proj(upsampled), h_masks)  # (B, T_feats, adim)

        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, 2, int(self.odim / 2)
        )  # (B, T_feats, odim)

        # after_outs = before_outs + self.postnet(
        #     before_outs.transpose(1, 2)
        # ).transpose(1, 2)

        before_loss, after_loss, duration_loss  = self.criterion(  #, prior_loss, spk_loss
            after_outs=None, #after_outs,
            before_outs=before_outs,
            # word_style_enc=word_style_enc,
            # prior_out=prior_out,
            d_outs=d_outs,
            ys=feats,
            ds=durations,
            ilens=ilens,
            olens=feats_lengths,
            sids=sids,
            # spk_class=spk_class,
            #spk_emb_preds=spk_emb_preds,
            #spk_embs=spembs
        )
        loss = duration_loss + before_loss.sum() #+ prior_loss + spk_loss
        
        if after_loss is not None:
            loss += after_loss

        stats = dict(
            before_loss=before_loss.sum().item(),
            duration_loss=duration_loss.item(),
            # prior_loss=prior_loss.item(),
            # spk_loss=spk_loss.item()
        )

        if after_loss is not None:
            stats["after_loss"]=after_loss.item() 

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
        sids:torch.Tensor,
        #pitch:torch.Tensor,
        #energy: torch.Tensor,
        #phone_word_mappings:torch.Tensor,
        #feats_word_avg: torch.Tensor,
        #spembs:torch.Tensor,
    ) -> Tuple[torch.Tensor,torch.Tensor]:

        return self._inference(text, 
                                sids=sids,
                                #pitch=pitch,energy=energy
                                #phone_word_mappings=phone_word_mappings,
                                #feats_word_avg=feats_word_avg)
                                #spembs=spembs
                                )     

    def inference(
        self,
        text: torch.LongTensor,
        durations:Optional[torch.Tensor] = None,
        pitch:Optional[torch.Tensor] = None,
        phone_word_mappings: Optional[torch.Tensor] = None,
        sids:torch.Tensor = None,
        feats_word_avg: Optional[torch.Tensor ]= None, 
        feats_word_avg_lengths: torch.Tensor = None,
        durations_lengths:Optional[torch.Tensor] = None,
        feats_lengths:Optional[torch.Tensor] = None,
        pitch_lengths:Optional[torch.Tensor] = None,
        feats: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        spembs: torch.Tensor = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feats, durations = self._inference(
            text=text, 
            durations=durations, 
            pitch=pitch, 
            phone_word_mappings=phone_word_mappings, 
            sids=sids, 
            feats_word_avg=feats_word_avg,
            durations_lengths=durations_lengths, 
            feats_lengths=feats_lengths, 
            pitch_lengths=pitch_lengths,
            energy=energy,
            # spembs=spembs,
            threshold=threshold, 
            minlenratio=minlenratio, 
            maxlenratio=maxlenratio, 
            use_att_constraint=use_att_constraint, 
            backward_window=backward_window, 
            forward_window=forward_window, 
            use_teacher_forcing=use_teacher_forcing)
        return dict(feat_gen=feats, durations=durations)
    
    def _inference(
        self,
        text: torch.Tensor,
        sids:torch.Tensor,
        phone_word_mappings:Optional[ torch.Tensor] = None,
        feats_word_avg: Optional[ torch.Tensor ] = None, 
        feats_word_avg_lengths: Optional[torch.Tensor] = None,
        durations:Optional[torch.Tensor] = None,
        pitch:Optional[torch.Tensor] = None,
        #spembs:torch.Tensor = None,
        durations_lengths:Optional[torch.Tensor] = None,
        feats_lengths:Optional[torch.Tensor] = None,
        pitch_lengths:Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,        
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        use_teacher_forcing: bool = False,
    ) -> Tuple [torch.Tensor,torch.Tensor]:
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
        print(text)
        
        # phone_word_mappings = phone_word_mappings.unsqueeze(0)       

        # feats_word_avg = feats_word_avg.unsqueeze(0)

        d_masks = make_pad_mask(ilens).to(text.device)
        
        # encode phone sequence
        phone_enc, _ = self.enc(text, ilens)

        # phone_enc = phone_enc.reshape(phone_word_mappings.size(0),phone_word_mappings.size(1),-1)

        # linear project
        # phone_enc_proj = self.word_seq_proj(phone_enc)
        # sids = torch.tensor([3]).to(sids.device)
        spembs = self.spk_embed(sids)#.unsqueeze(0)

        # average by word boundaries
        # phone_enc_averaged = self.average(phone_enc_proj, phone_word_mappings, d_masks).view(1, phone_word_mappings.max()+1, -1)

        # word_enc_out, _ = self.word_seq_enc(phone_enc_averaged)

        # prior_out = self.prior(phone_enc_averaged, word_enc_out)  

        # word_style_enc = self.word_style_encoder(feats_word_avg)

        # prior_out += (0.05 * word_style_enc.sum(1))

        #for i in range(15):
        #    style_token = int(os.environ["STYLE_TOKEN"])
        # style_weighting = float(os.environ["STYLE_WEIGHTING"])
        # prior_out += (style_weighting * self.word_style_encoder.style_tokens[:,style_token,:].unsqueeze(1).expand(prior_out.size()))
        # prior_out += 0.5 * self.word_style_encoder.style_tokens[:,5,:].unsqueeze(1).expand(prior_out.size())

        spembs = spembs.unsqueeze(0)

        # concatenated = self.concatenate(word_enc_out, phone_enc_proj, phone_word_mappings, \
        # # prior_out, 
        # word_style_enc,
        # spembs)
        # concatenated = concatenated.view(phone_enc_proj.size(0), phone_enc_proj.size(1), self.decoder_adim)
        # concatenated = concatenated.reshape(phone_enc_proj.size(0), phone_enc_proj.size(1), self.decoder_adim)
        # pitch = torch.tensor([4] * phone_enc.size(1)).to(sids.device)
        # energy = torch.tensor([4] * phone_enc.size(1)).to(sids.device)

        # pitch_embs = self.pitch_embed(pitch).unsqueeze(0)
        # energy_embs = self.energy_embed(energy).unsqueeze(0)

        # print(phone_enc.size())
        # print(spembs.size())
        # print(pitch_embs.size())
        # print(energy_embs.size())

        concatenated = phone_enc #torch.cat([
            #phone_enc, 
            #spembs.expand(-1,phone_enc.size(1), -1), 
            #pitch_embs, 
            #energy_embs,
        #], 2)

        # forward duration predictor and variance predictors        
        d_outs = self.duration_predictor.inference(concatenated, d_masks)
        r_outs = self.range_predictor.inference(concatenated, d_masks)

        # apply Gaussian upsampling
        upsampled = self.length_regulator(concatenated, r_outs, d_outs) #durations.unsqueeze(0))  # (B, T_feats, adim)
        #upsampled = self.length_regulator(concatenated, torch.zeros(r_outs.size()), durations.unsqueeze(0))  # (B, T_feats, adim)
        
        h_masks = torch.ones(1, upsampled.size(1), dtype=torch.bool).to(concatenated.device)
        
        zs, _ = self.dec(self.enc_proj(upsampled),h_masks)  # (B, T_feats, adim)

        # if self.proj_up_down:
        #     zs = self.dec_proj_down(self.dec_proj_up(zs))
                
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, 2, int(self.odim / 2)
        ).argmax(-1)  # (B, T_feats, odim)

        after_outs = before_outs #+ self.postnet(
            #before_outs.transpose(1, 2)
        #).transpose(1, 2)
        
        return after_outs, d_outs
        
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
