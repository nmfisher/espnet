from pynini import escape
import six

import math
import torch
import torch.nn.functional as F
import numpy as np 
from espnet.nets.pytorch_backend.hqss.prenet import Prenet

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device

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

        attn_init(self)

    
    def reset(self):
        """reset states"""
        self.means = None
        self.enc_i_pos = None
        self.rnn_h = None
        self.iteration+=1 
        self.decoding_step = 0 
   
    def forward(
        self,
        enc_z,
        enc_z_lens,
        rnn_c,
        # rnn_h,
        #last_out
        prev_att_w,
        prev_c
    ):
        """Calculate AttLoc forward propagation.
        :param torch.Tensor enc_z: encoder hidden states
        :return: concatenated context (B x N x D_dec)
        :rtype: torch.Tensor
        """
        #torch.autograd.set_detect_anomaly(True)
        batch = len(enc_z)
        device = enc_z.device

        if self.enc_i_pos is None:
          # rnn_c = torch.zeros(rnn_c.size(0), self.adim).to(device)
          isteps = enc_z.size(1)
          irange = torch.tensor(
            [
              np.arange(
                isteps
              )
            ]
          )
          irange = irange.unsqueeze(-1).expand(
              batch,
              isteps,
              self.num_dists
            ).to(device)
          
          self.enc_i_pos = irange + 0.5
          self.enc_i_neg = irange - 0.5
          
          self.pad_mask = make_pad_mask(enc_z_lens)

        # rnn_c = torch.cat([rnn_c, prev_c], dim=1).unsqueeze(1)
        rnn_c = rnn_c.unsqueeze(1)

        rnn_c, self.rnn_h = self.rnn(
           rnn_c,self.rnn_h
        )        

        params = self.param_net(rnn_c)

        means = F.sigmoid(
            params[:,:,0:self.num_dists]
        ) / 5

        if self.means is not None:
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

        if self.iteration % 10 == 0:
          print(f"decoding step {self.decoding_step} enc_z {enc_z.size()} m {means[0,0]} s {scales[0,0]} w {weights[0,0]} f {f[0,0]} rnn_w {rnn_w[0][0]}")

        rnn_w += 1e-7
        rnn_w[self.pad_mask] = 0
        # rnn_w[self.pad_mask] = -math.inf
        # rnn_w = F.softmax(rnn_w, 1)

        self.decoding_step += 1

        # if self.iteration % 25 == 0:
          # print(f"m {means[0,0]} s {scales[0,0]} w {weights[0,0]} f {f[0,0]} rnn_w {rnn_w[0]}")
          # print(rnn_w[0])
          # print(rnn_w[1])
        # prod = rnn_w * enc_z
        # print(f"w {rnn_w.size()} e {enc_z.size()} p {prod.size()} {rnn_w}")
        
        
        # print(torch.sum(
        #   prod,
        #   dim=1
        # ).size())

        # raise Exception()

        return torch.sum(
          rnn_w * enc_z, 
          dim=1
        ).to(device),  \
        rnn_w.squeeze(-1) #, rnn_h  
        
        


          # print(f"mean {mean.size()} encipos {self.enc_i_pos.size()} scale {scale.size()} f1 {f1.size()}")
          # print(f"mean {mean.size()} encipos {self.enc_i_pos.size()} scale {scale.size()} f1 {f1.size()}")

                  #print(f" means {len(self.means)} {self.means[-1][0][0]}")
        #print(f" weights {weights.size()} {weights}")
        #print(f" f {f.size()} {f}")
        # print(weights.size())       
        # print(f.size())
                  # print(f"f1 {f1.size()} f2 {f2.size()}")