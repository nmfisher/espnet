import six

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.hqss.prenet import Prenet


class MOLAttn(torch.nn.Module):
    """Mixture-of-logistics RNN attention module.
    At each timestep [i], accepts a Tensor of size B x 1 x input_dim
    Returns a Tensor
    :param int input_dim: dimension of encoder outputs (also dim of context vector)
    :param int output_dim: dimension of decoder pre-net outputs (RNN attention layer then accept a tensor of dim (enc_odim + cdim)
    :param int att_dim: hidden size for RNN attention layer
    :param int num_dist: number of mixture components to use
    """

    def __init__(
        self, input_dim, output_dim, att_dim, num_dists=5, frames_per_step=5
    ):
        super(MOLAttn, self).__init__()
        
        self.input_dim = input_dim
        self.att_dim = att_dim
        self.output_dim = output_dim
        self.frames_per_step = frames_per_step
        self.num_dists = num_dists
        self.rnn_input_dim = 512

        self.rnn = torch.nn.GRU(self.rnn_input_dim, att_dim, batch_first=True)

        
        # fully-connected layer for predicting logistic distribution parameters
        # mean/scale/mixture weight
        # accepts hidden state of B x S x HS
        # outputs B x S x 
        self.param_nets = [ torch.nn.Sequential(
                torch.nn.Linear(self.att_dim, 256), 
                torch.nn.Tanh(),
                torch.nn.Linear(256, 3), 
            ) for k in range(self.num_dists) 
        ]        

    def _logistic(self, x, mean, scale):
        return (1 / (1 + torch.exp((x - mean) / scale))) + 1e-6

    def forward(
        self,
        input,
        enc_seq_len=None,
        device=None,
        reset=False
    ):
        """Calculate AttLoc forward propagation.
        :param torch.Tensor enc_z: encoder outputs (B x N x D_dec)
        :param torch.Tensor dec_z: last acoustic frame output (processed by decoder pre-net) (B x  D_dec)
        :param int i: decoder step        
        :return: concatenated context + hidden state (B x N x D_dec)
        :rtype: torch.Tensor
        """
        torch.autograd.set_detect_anomaly(True)
        if reset:
          self.means = []
        #print(f"input {input.size()}")        
        out, self.state = self.rnn(input.to(device), None if reset else self.state)
        
        #print(f"input {input.size()} out {out.size()} selfstate {self.state.size()}")

        # create placeholder for output i.e. (concatenated context + hidden state) that will be fed to downstream decoders
        # B x N x (DX2) (where D is the dimension of the context/hidden vectors)
        
        # pass input and hidden state to RNN
        # returns new context vector and hidden state for this timestep
        # during training, the input on the first decoder step will simply be a zero frame
        # subequent steps will use the nth decoder pre-net output frame  (i.e. acoustic features passed through pre-net) with the last context vector
        # during inference time, this is replaced with the predicted acoustic features passed through prenet
      
        weights = []
        scales = []

        means = []
        # apply FC nets to get params for each mixture            
        for k in range(self.num_dists):
            param_net = self.param_nets[k].to(device)
            params = param_net(torch.squeeze(self.state, 0))
            
            if reset:
              means += [ torch.exp(params[:,0]) ]
            else:              
              means += [ torch.exp(params[:,0]) + self.means[-1][k]  ]
            
            scales += [ torch.exp(params[:,1]) ]
            
            weights += params[:, 2]
        self.means += [ means ]
        weights = torch.nn.functional.softmax(torch.Tensor(weights), dim=0)
        
        # container for encoder alignment probabilities
        a = []
        
        # calculate alignment probabilities for each encoder timestep
        for j in range(enc_seq_len):
          a += [ 0 ]
          for k in range(self.num_dists):
            mean = means[k]
            scale = scales[k]

            f1 = self._logistic(j + 0.5,  mean, scale)
            f2 = self._logistic(j - 0.5, mean, scale)

            step_prob = weights[k] * (f1 - f2)
            
            a[-1] += step_prob
        probs = torch.cat(a,dim=-1)
        
        probs = probs.reshape(input.size()[0], enc_seq_len, 1).to(device)
        return probs, self.state, weights


