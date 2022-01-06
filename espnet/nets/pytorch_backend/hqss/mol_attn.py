import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.hqss.prenet import Prenet


class MOLAttn(torch.nn.Module):
    """mixture-of-logistics attention module.
    :param int enc_odim: dimension of encoder outputs (also dim of context vector)
    :param int dec_odim: dimension of decoder pre-net outputs (RNN attention layer then accept a tensor of dim (enc_odim + cdim)
    :param int att_dim: hidden size for RNN attention layer
    :param int num_dist: number of mixture components to use
    """

    def __init__(
        self, enc_odim, dec_odim, att_dim, batch_size, num_dists=5, frames_per_step=5
    ):
        super(MOLAttn, self).__init__()
        
        self.rnn_input_dim = enc_odim + dec_odim
        self.rnn = torch.nn.GRU(self.rnn_input_dim, att_dim, batch_first=True)

        self.enc_odim = enc_odim
        self.att_dim = att_dim
        self.dec_odim = dec_odim
        self.frames_per_step = frames_per_step
        self.num_dists = num_dists
        self.batch_size = batch_size
        
        # fully-connected layer for predicting logistic distribution parameters
        # mean/scale/mixture weight
        # accepts hidden state of B x S x HS
        # outputs B x S x 
        self.param_nets = [ torch.nn.Sequential(
                torch.nn.Linear(self.att_dim, self.att_dim), 
                torch.nn.Tanh(),
                torch.nn.Linear(self.att_dim, 3), 
            ) for k in range(self.num_dists) 
        ]        

    def _logistic(self, x, mean, scale):
        return 1 / (1 + torch.exp(-(x - mean / scale)))

    def _reset(self, device):
        # randomize initial hidden state (? check this)
        self.h = torch.randn(1, self.batch_size, self.att_dim, device=device)
        self.c = torch.zeros(self.batch_size, self.enc_odim, device=device)
        # create placeholder for means of logistic distributions
        self.means = torch.zeros(self.batch_size, self.num_dists, device=device)

    def forward(
        self,
        enc_z,
        dec_z,
        i
        #,
        #y,
    ):
        """Calculate AttLoc forward propagation.
        :param torch.Tensor enc_z: encoder outputs (B x N x D_dec)
        :param torch.Tensor dec_z: last acoustic frame output (processed by decoder pre-net) (B x  D_dec)
        :param int i: decoder step        
        :return: concatenated context + hidden state (B x N x D_dec)
        :rtype: torch.Tensor
        """
        device = enc_z.device
        if i == 0:
            self._reset(device)

        enc_seq_len = enc_z.size()[1]

        # create placeholder for output i.e. (concatenated context + hidden state) that will be fed to downstream decoders
        # B x N x (DX2) (where D is the dimension of the context/hidden vectors)
        
        # pass input and hidden state to RNN
        # returns new context vector and hidden state for this timestep
        # during training, the input on the first decoder step will simply be a zero frame
        # subequent steps will use the nth decoder pre-net output frame  (i.e. acoustic features passed through pre-net) with the last context vector
        # during inference time, this is replaced with the predicted acoustic features passed through prenet
        
        # concatenate pre-net output with last context vector
        #print(dec_z.size())
        #print(self.c.size())
        att_in = torch.unsqueeze(torch.cat([dec_z, self.c], dim=1), dim=1).to(device)

        # pass input and last state into RNN
        o_i,self.h = self.rnn(att_in, self.h)

        weights = torch.zeros((self.batch_size, self.num_dists), device=device)

        # apply FC nets to get params for each mixture            
        for k in range(self.num_dists):
            param_net = self.param_nets[k].to(device)
            params = param_net(self.h)
            
            mean = self.means[:, k] + torch.exp(params[0,:,0])
            scale = torch.exp(params[0, :,1])
            weights[:,k] = params[0, :, 2]
            
        weights = torch.nn.functional.softmax(weights, dim=1)

        torch.autograd.set_detect_anomaly(True)
        
        # placeholder tensor for encoder timestep probabilities
        a = torch.zeros(self.batch_size, enc_seq_len).to(device)
        
        for k in range(self.num_dists):
            # calculate alignment probabilities for each encoder timestep
            for j in range(enc_seq_len):
                f1 = self._logistic(j + 0.5,  mean, scale)
                f2 = self._logistic(j - 0.5, mean, scale)

                a[:,j] += weights[:,k]  * (f1 - f2)
        
        self.c = torch.sum(torch.unsqueeze(a,2).to(device) * enc_z, 1).to(device)

        # return context vector & hidden state               
        return torch.unsqueeze(self.c, 0).to(device), self.h


