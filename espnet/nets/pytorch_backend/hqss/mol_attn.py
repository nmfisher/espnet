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
        self, enc_odim, dec_odim, att_dim, num_dists=5, frames_per_step=5
    ):
        super(MOLAttn, self).__init__()
        
        self.rnn_input_dim = enc_odim + dec_odim
        self.rnn = torch.nn.GRU(self.rnn_input_dim, att_dim, batch_first=True)

        self.enc_odim = enc_odim
        self.att_dim = att_dim
        self.dec_odim = dec_odim
        self.frames_per_step = frames_per_step
        self.num_dists = num_dists
        
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
        return (1 / (1 + torch.exp((x - mean) / scale))) + 1e-6

    def _reset(self, device, batch_size):
        # randomize initial hidden state (? check this)
        self.h = torch.randn(1, batch_size, self.att_dim, device=device, requires_grad=True)
        self.c = torch.zeros(batch_size, self.enc_odim, device=device, requires_grad=True)
        # create placeholder for means of logistic distributions
        self.means = []

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
        torch.autograd.set_detect_anomaly(True)

        device = enc_z.device
        batch_size = enc_z.size()[0]
        enc_seq_len = enc_z.size()[1]
        
        if i == 0:
            self._reset(device, batch_size)        

        # create placeholder for output i.e. (concatenated context + hidden state) that will be fed to downstream decoders
        # B x N x (DX2) (where D is the dimension of the context/hidden vectors)
        
        # pass input and hidden state to RNN
        # returns new context vector and hidden state for this timestep
        # during training, the input on the first decoder step will simply be a zero frame
        # subequent steps will use the nth decoder pre-net output frame  (i.e. acoustic features passed through pre-net) with the last context vector
        # during inference time, this is replaced with the predicted acoustic features passed through prenet
        
        # concatenate pre-net output with last context vector
        
        att_in = torch.unsqueeze(torch.cat([dec_z, self.c], dim=1), dim=1).to(device)

        # pass input and last state into RNN
        o_i,self.h = self.rnn(att_in, self.h)

        weights = []
        scales = []
        dec_step_means = []

        # apply FC nets to get params for each mixture            
        for k in range(self.num_dists):
            param_net = self.param_nets[k].to(device)
            params = param_net(self.h)
            
            dec_step_means += [ torch.exp(params[0,:,0]) + self.means[-1][k] if i > 0 else torch.exp(params[0,:,0]) ]
            
            scales += [ torch.exp(params[0,:,1]) ]
            
            weights += params[0,:, 2]
        
        self.means += [ dec_step_means ]
        weights = torch.Tensor(weights)
        
        weights = torch.nn.functional.softmax(torch.Tensor(weights), dim=0)

        
        # placeholder tensor for encoder timestep probabilities
        a = torch.zeros(batch_size, enc_seq_len, requires_grad=True).to(device)
        
        for k in range(self.num_dists):
            mean = self.means[-1][k]
            scale = scales[k]
            weight = weights[k]
            
            # calculate alignment probabilities for each encoder timestep
            for j in range(enc_seq_len):
                f1 = self._logistic(j + 0.5,  mean, scale)
                f2 = self._logistic(j - 0.5, mean, scale)

                a[:,j] += weights[k] * (f1 - f2)
        a = torch.unsqueeze(a,2)
        
        self.c = torch.sum(a * enc_z, 1).to(device)

        # return context vector & hidden state               
        return torch.unsqueeze(self.c, 0).to(device), self.h, weights


