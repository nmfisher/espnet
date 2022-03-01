from pynini import escape
import six

import math
import torch
import torch.nn.functional as F
import numpy as np 
from espnet.nets.pytorch_backend.hqss.prenet import Prenet

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device

from typing import Optional

def attn_init(m):
    """Initialize decoder parameters."""
    return
    #torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("tanh"))


class MOLAttn(torch.nn.Module):
    """Mixture-of-logistics RNN attention module.
    At each timestep [i], accepts a Tensor of size B x 1 x input_dim
    Returns a Tensor
    :param int output_dim: dimension of decoder pre-net outputs (RNN attention layer then accept a tensor of dim (enc_odim + cdim)
    :param int num_dist: number of mixture components to use
    """

    def __init__(
        self, enc_dim, adim, num_dists=5
    ):
        super(MOLAttn, self).__init__()
        self.enc_dim = enc_dim
        self.adim = adim
        self.rnn = torch.nn.GRU(enc_dim, adim, batch_first=True)
        
        self.num_dists = num_dists
        
        # fully-connected layer for predicting logistic distribution parameters
        # mean/scale/mixture weight
        # accepts hidden state of B x S x HS
        # outputs B x S x 
        self.param_net = torch.nn.Sequential(
                torch.nn.Linear(adim, 128), 
                torch.nn.Tanh(),
                torch.nn.Linear(128, self.num_dists * 3)
            ) 

        self.iteration = 0

        MAX_ISTEPS = 200

        self.isteps = torch.tensor(
            [
              list(range(
                MAX_ISTEPS
              ))
            ]
          ).unsqueeze(1)

        self.enc_i_pos : Optional [ torch.Tensor ] = None
        self.enc_i_neg : Optional [ torch.Tensor ] = None
        self.pad_mask : Optional [ torch.Tensor ] = None
        self.means : Optional [ torch.Tensor ] = None
        self.rnn_h : Optional [ torch.Tensor ] = None
        self.reset()

        attn_init(self)

    
    def reset(self):
        """reset states"""
        self.iteration += 1 
        self.means =  torch.zeros(0, dtype=torch.float32)
        self.enc_i_pos : Optional [ torch.Tensor ] = torch.zeros(0, dtype=torch.float32)
        self.enc_i_neg : Optional [ torch.Tensor ] = torch.zeros(0, dtype=torch.float32)
        self.pad_mask : Optional [ torch.Tensor ] = torch.ones(0, dtype=torch.bool)
        self.rnn_h : Optional [ torch.Tensor ] = torch.zeros(0, dtype=torch.float32)
        self.iteration+=1 
        self.decoding_step = 0 
   
    def forward(
        self,
        enc_z,
        enc_z_lens,
        rnn_c,
        # rnn_h,
        #last_out
        prev_att_w : Optional [ torch.Tensor ] = None,
        prev_c : Optional [ torch.Tensor ] = None
    ):
        """Calculate AttLoc forward propagation.
        :param torch.Tensor enc_z: encoder hidden states
        :return: concatenated context (B x N x D_dec)
        :rtype: torch.Tensor
        """
        #torch.autograd.set_detect_anomaly(True)
        batch = len(enc_z)
        device = enc_z.device

        if self.decoding_step == 0:
          self.irange = self.isteps[:,:,:enc_z.size(1)].expand(
              batch,
              self.num_dists,
              enc_z.size(1)
             ).to(device).transpose(2,1)
          # isteps = enc_z.size(1)
          # irange = torch.tensor(
          #   [
          #     np.arange(
          #       isteps
          #     )
          #   ]
          # )
          # self.irange = irange.unsqueeze(-1).expand(
          #     batch,
          #     isteps,
          #     self.num_dists
          #   ).to(device)
          
          self.enc_i_pos = self.irange + 0.5
          self.enc_i_neg = self.irange - 0.5
          
          self.pad_mask = make_pad_mask(enc_z_lens)

        # rnn_c = torch.cat([rnn_c, prev_c], dim=1).unsqueeze(1)
        rnn_c = rnn_c.unsqueeze(1)

        if self.decoding_step > 0:
          rnn_c, self.rnn_h = self.rnn(
            rnn_c,self.rnn_h
          )        
        else:
          rnn_c, self.rnn_h = self.rnn(
            rnn_c,None
          )        
 

        params = self.param_net(rnn_c)

        means = F.sigmoid(
            params[:,:,0:self.num_dists]
        ) / 5

        if self.decoding_step > 0:
          means = means + self.means

        self.means = torch.clamp(
          means,
          min=0, max=enc_z.size(1)
        )
        
        scales = params[:,:,self.num_dists:self.num_dists*2]
        # scales[scales < -7] = -7
        scales = torch.exp(scales)
          
        weights = F.softmax(
          params[:,:,self.num_dists*2:]
          , -1)

        # print(f"means {means.size()} encipos {self.enc_i_pos.size()} scales {scales.size()} weights {weights.size()}  self.enc_i_pos{self.enc_i_pos.size()}")
        # print(self.enc_i_pos)

        means = self.means.expand(self.enc_i_pos.size())
        scales = scales.expand(self.enc_i_pos.size())
        weights = weights.expand(self.enc_i_pos.size())
        
        f1 = F.sigmoid((self.enc_i_pos - means) / scales)

        f2 = F.sigmoid((self.enc_i_neg - means) / scales)

        f = f1 - f2
                
        rnn_w = (weights * f)
        
        rnn_w = rnn_w.sum(2).unsqueeze(-1)

        rnn_w += 1e-7
        rnn_w[self.pad_mask] = 0
        # rnn_w[self.pad_mask] = -math.inf
        # rnn_w = F.softmax(rnn_w, 1)

        self.decoding_step += 1

        return torch.sum(
          rnn_w * enc_z, 
          dim=1
        ).to(device),  \
        rnn_w.squeeze(-1) #, rnn_h  
        
        