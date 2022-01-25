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
        self.rnn = torch.nn.GRU(enc_dim + adim, adim, batch_first=True)
        
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
        # self.mean_net = torch.nn.Sequential(
        #   torch.nn.Linear(adim, adim * 2), 
        #   torch.nn.Tanh(),
        #   torch.nn.Linear(adim * 2, 1)
        # ) 

        # self.scale_net = torch.nn.Sequential(
        #   torch.nn.Linear(adim, adim * 2), 
        #   torch.nn.Tanh(),
        #   torch.nn.Linear(adim * 2, 1)
        # ) 

        # self.weight_net = torch.nn.Sequential(
        #   torch.nn.Linear(adim, adim * 2), 
        #   torch.nn.Tanh(),
        #   torch.nn.Linear(adim * 2, 1)
        # ) 

        self.dec_proj = torch.nn.Linear(dec_dim, adim, bias=False)

        attn_init(self)

    
    def reset(self):
        """reset states"""
        self.means = []
        self.out = None
        self.context = None
        self.state = None
        self.enc_indices = None
   
    def forward(
        self,
        enc_z,
        ilens,
        dec_step,
        att_prev
    ):
        """Calculate AttLoc forward propagation.
        :param torch.Tensor input: last acoustic frame
        :param int i: decoder step        
        :return: concatenated context + hidden state (B x N x D_dec)
        :rtype: torch.Tensor
        """
        #torch.autograd.set_detect_anomaly(True)
        batch = len(enc_z)
        device = enc_z.device

        if self.out is None:
          self.out = enc_z.new_zeros(
            enc_z.size(0), 
            1,
            self.adim
          )
          self.context = enc_z.new_zeros(
            enc_z.size(0), 
            1,
            self.enc_dim
          )
          self.enc_indices_pos = torch.FloatTensor(np.arange(enc_z.size(1))).to(device) + 0.5
          self.enc_indices_neg = torch.FloatTensor(np.arange(enc_z.size(1))).to(device) - 0.5
        
        rnn_in = torch.cat([
          self.out, 
          self.context.view(
            enc_z.size(0), 
            1,
            self.enc_dim
          )
        ], 2)
        
        out, self.state = self.rnn(
           rnn_in,
           self.state
        )

        self.out = out # torch.tanh(out + self.dec_proj(dec_step))

        params = torch.exp(self.param_net(self.out))

        alignment_probs = []
        all_means = []
        for j in range(self.num_dists):

          if len(self.means) > 0:
            means = params[:,:,j*3]  + self.means[-1][j]
            # means = torch.exp(self.mean_net(self.out))  + self.means[-1]
          else:
            means = params[:,:,j*3]

          all_means += [ means ]

          # scales = torch.exp(self.scale_net(self.out)) 
          scales = params[:,:,(j*3)+1]

          # weights = F.softmax(torch.exp(self.weight_net(self.out)), dim=1)
          weights = F.softmax(params[:,:,(j*3)+2], dim=1)
          
          f1 = F.sigmoid((self.enc_indices_pos - means) / scales)
          f2 = F.sigmoid((self.enc_indices_neg - means) / scales)

          alignment_probs += [ weights * (f1 - f2) ]
        self.means += [ all_means ]
        #print(alignment_probs[-1].size())
        alignment_probs = torch.stack(alignment_probs, 1)
        
        alignment_probs = torch.sum(alignment_probs, 1).unsqueeze(2)
        
        self.context = torch.sum(
          alignment_probs * enc_z, 
          dim=1
        ).to(device)

        return self.context, alignment_probs
        
        