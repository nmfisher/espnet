from pynini import escape
import six

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
        self, enc_dim, dec_dim, adim, num_dists=1, frames_per_step=5
    ):
        super(MOLAttn, self).__init__()
        self.enc_dim = enc_dim
        self.adim = adim
        self.rnn = torch.nn.GRU(adim + enc_dim, adim, batch_first=True)
        
        self.num_dists = num_dists
        
        # fully-connected layer for predicting logistic distribution parameters
        # mean/scale/mixture weight
        # accepts hidden state of B x S x HS
        # outputs B x S x 
        self.param_net = torch.nn.Sequential(
                torch.nn.Linear(adim, adim), 
                torch.nn.Tanh(),
                torch.nn.Linear(adim, self.num_dists * 3)
            ) 

        attn_init(self)

    
    def reset(self):
        """reset states"""
        self.means = []
        self.enc_i = None
   
    def forward(
        self,
        enc_z,
        rnn_c,
        rnn_h,
        last_out
    ):
        """Calculate AttLoc forward propagation.
        :param torch.Tensor enc_z: encoder hidden states
        :return: concatenated context (B x N x D_dec)
        :rtype: torch.Tensor
        """
        #torch.autograd.set_detect_anomaly(True)
        batch = len(enc_z)
        device = enc_z.device

        if rnn_c is None:
          rnn_c = enc_z.new_zeros(
            enc_z.size(0), 
            1,
            #self.enc_dim + self.adim
            276
          )
          self.enc_i_pos = torch.FloatTensor(np.arange(enc_z.size(1))).unsqueeze(0).expand(enc_z.size(0), enc_z.size(1)).to(device) + 0.5
          self.enc_i_neg = torch.FloatTensor(np.arange(enc_z.size(1))).unsqueeze(0).expand(enc_z.size(0), enc_z.size(1)).to(device) - 0.5
        else:
          rnn_c = torch.cat([rnn_c, last_out], dim=1).unsqueeze(1)
          assert rnn_h is not None
        
        rnn_c, rnn_h = self.rnn(
           rnn_c,rnn_h
        )        

        params = self.param_net(rnn_c)
        
        rnn_w = []
        self.means += [[]]
        weights = []
        f1 = []
        f2 = []
        for k in range(self.num_dists):

          if len(self.means) > 1:
            mean = torch.exp(params[:,:,k*3])  + self.means[-2][k]
          else:
            mean = torch.exp(params[:,:,k*3])
          self.means[-1] += [ mean ]

          scale = torch.exp(params[:,:,(k*3)+1])
          
          weights += [ params[:,:,(k*3)+2] ]
          
          s1 = F.sigmoid((self.enc_i_pos - mean) / scale).unsqueeze(1)
          
          f1 += [ s1 ] 
          
          f2 += [ F.sigmoid((self.enc_i_neg - mean) / scale).unsqueeze(1) ]
       
        f1 = torch.cat(f1, 1)
        
        f2 = torch.cat(f2, 1)

        weights = torch.cat(weights, 1)
        
        weights = F.softmax(weights, 1).unsqueeze(-1)
        
        weights = weights.expand(batch, weights.size(1), f1.size(2))
        
        rnn_w = (weights * (f1 - f2))
                
        rnn_w = rnn_w.sum(1).unsqueeze(-1)
        
        rnn_c = torch.sum(
          rnn_w * enc_z, 
          dim=1
        ).to(device)

        return rnn_c, rnn_w.squeeze(-1), rnn_h  
        
        