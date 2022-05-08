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

import torch
import torch.nn.functional as F
import time

from typeguard import check_argument_types
#from espnet.nets.pytorch_backend.hqss.loc_attn import AttLoc
from espnet.nets.pytorch_backend.rnn.attentions2 import AttLoc
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder
from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder

from espnet2.tts.abs_tts import AbsTTS

from espnet.nets.pytorch_backend.hqss.postnet import Postnet
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.tacotron2.utils import make_non_pad_mask
from espnet.nets.pytorch_backend.tacotron2.utils import make_pad_mask

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.fastspeech2.variance_predictor import VariancePredictor

from espnet2.tts.wlsc.loss import WLSCLoss

class WordStyleEncoder(torch.nn.Module):
    def __init__(self,idim, gru_units, gru_layers, num_style_tokens=5, style_token_dim=128, num_attn_heads=4):
        super().__init__()
        self.num_style_tokens = num_style_tokens
        self.style_tokens = torch.nn.Parameter(torch.randn(1, self.num_style_tokens,style_token_dim))
        self.gru = torch.nn.GRU(idim, gru_units, gru_layers, batch_first=True)
        self.gru_units = gru_units
        self.gru.flatten_parameters()
        self.gru_layers = gru_layers
        self.attn = torch.nn.MultiheadAttention(gru_units,num_attn_heads, kdim=style_token_dim, vdim=style_token_dim, batch_first=True)

    # feats_averaged is BxLwxF
    # phone_word_mappings is BxLp
    # durations is BxLxp
    # (B=batch size, Lw = word sequence length (i.e. num words), La= acoustic length (i.e. num frames), F=feature dimension, Lp = phonetic sequence length)
    def forward(self, feats_averaged:torch.Tensor):
        # each element in outs represents an encoding of the acoustic features for a word (being the averaged acoustic features for each phone)
        outs, _ = self.gru(feats_averaged)
        
        st = self.style_tokens.expand(outs.size(0), -1, -1)

        encoded = []
        # MHA doesn't accept 3D inputs?
        for i in range(outs.size(1)):
          word = outs[:,i].unsqueeze(1) # this is the i'th word for all sequences in the batch       
          attn_out, attn_w = self.attn(word,st,st)
          encoded += [attn_out]
        return torch.cat(encoded, 1)

class WLSC(AbsTTS):
    """WLSC module for end-to-end text-to-speech."""
    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        num_speakers: Optional[int],
        spks:Optional[int],
    ):
        """Initialize WLSC module.
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
                
        # set padding idx
        self.padding_idx = 0

        self.embed_dim = 128

        # define network modules
        self.enc = Encoder(
            idim=idim,
            embed_dim=self.embed_dim,
            dropout_rate=0.5,
            padding_idx=self.padding_idx,
        )
        self.word_seq_proj_dim = 128
        self.word_seq_proj = torch.nn.Linear(512, self.word_seq_proj_dim)
        self.word_style_encoder = WordStyleEncoder(self.odim,self.embed_dim, 2)

        self.dec = Decoder(
            768,
            odim,
            AttLoc(768, 1024, odim, 32, 15)
        )

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=768, # concatenated size
            n_layers=2,
            n_chans=256,
            kernel_size=3,
            dropout_rate=0.5,
        )

        # define length regulator
        self.length_regulator = LengthRegulator()
        
        self.feat_out = torch.nn.Linear(128, odim)

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
        # zero out the masked entries so we're not inadvertently mistaking padding as word at index 0
        # print(mask.size())
        # print(mask)
        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, word_phone_mappings.size(1), -1)
        word_phone_mappings[mask] = 0
        product = torch.matmul(word_phone_mappings, embeddings)
        return product / word_phone_mappings.sum(2).unsqueeze(-1)
    
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
        word_style_embeddings_spread = torch.gather(word_style_embeddings, 1, phone_word_mappings.unsqueeze(-1).expand(phone_word_mappings.size(0), phone_word_mappings.size(1), word_embeddings.size(2)))
        return torch.cat([word_embeddings_spread, phone_embeddings,word_style_embeddings_spread], -1)

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

        # encode phone sequence
        phone_enc, _ = self.enc(text, ilens)

        # linear project
        phone_enc_proj = self.word_seq_proj(phone_enc)

        # average by word boundaries
        word_enc = self.average(phone_enc_proj, phone_word_mappings, d_masks)

        print(word_enc)

        concatenated = self.concatenate(word_enc, phone_enc_proj, phone_word_mappings, word_style_enc)

        # forward duration predictor and variance predictors
        
        d_outs = self.duration_predictor(concatenated, d_masks)
                    
        concatenated = self.length_regulator(concatenated, durations)  # (B, T_feats, adim)
        is_inference = False
        # forward decoder
        if feats_lengths is not None and not is_inference:
            h_masks = self._source_mask(feats_lengths)
        else:
            h_masks = torch.ones(1, concatenated.size(1), dtype=torch.bool).to(concatenated.device)
        
        after_outs, before_outs, logits, att_ws = self.dec(concatenated, feats_lengths, feats)  # (B, T_feats, adim)

        l1_loss, duration_loss = self.criterion(
            after_outs=None,
            before_outs=before_outs,
            d_outs=d_outs,
            ys=feats,
            ds=durations,
            ilens=ilens,
            olens=feats_lengths,
        )
        loss = l1_loss + duration_loss  

        stats = dict(
            l1_loss=l1_loss.item(),
            duration_loss=duration_loss.item(),
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

        # get speaker embeddings
        spkr_emb = self.spk_embd(sids)

        # encode phone sequence
        phone_enc, _ = self.enc(text, durations, pitch, ilens)

        outs,probs, phone_att_ws = self.dec.inference(phone_enc, ilens, spkr_emb)
        
        return dict(feat_gen=outs.squeeze(0).transpose(0,1), prob=None, att_w=phone_att_ws)
        

    
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