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
        self, input_dim, output_dim, att_dim, num_dists=1, frames_per_step=5
    ):
        super(MOLAttn, self).__init__()
        
        self.input_dim = input_dim
        self.att_dim = att_dim
        self.output_dim = output_dim
        self.frames_per_step = frames_per_step
        self.num_dists = num_dists



        self.proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
        )
        
        # fully-connected layer for predicting logistic distribution parameters
        # mean/scale/mixture weight
        # accepts hidden state of B x S x HS
        # outputs B x S x 
        self.param_net = torch.nn.Sequential(
                torch.nn.Linear(512, 256), 
                torch.nn.Tanh(),
                torch.nn.Linear(256, self.num_dists  * 3)
            ) 
   
    def _logistic(self, x, mean, scale):
      return torch.nn.functional.sigmoid((x - mean) / scale)

    def forward(
        self,
        state,
        enc_z,
        device=None,
        reset=False
    ):
        """Calculate AttLoc forward propagation.
        :param torch.Tensor input: last acoustic frame
        :param int i: decoder step        
        :return: concatenated context + hidden state (B x N x D_dec)
        :rtype: torch.Tensor
        """
        torch.autograd.set_detect_anomaly(True)

        # applies FFN to get MoL params
        # returned tensor will be B x (num_dist * 3), so we reshape to B x num_dist x 3
        # 0 is mean, 1 is scale, 2 is weight
        params = torch.exp(self.param_net(state)) 

        mean_indices = torch.LongTensor([(i*3) for i in range(self.num_dists)]).to(device)
        mean_indices = torch.tile(mean_indices, [params.size()[0], 1, 1])
        means = torch.gather(params, 2, mean_indices)
        
        scale_indices = torch.LongTensor([1+(i*3) for i in range(self.num_dists)]).to(device)
        scale_indices = torch.tile(scale_indices, [params.size()[0], 1, 1])
        scales = torch.gather(params, 2, scale_indices)

        weights_indices = torch.LongTensor([2+(i*3) for i in range(self.num_dists)]).to(device)
        weights_indices = torch.tile(weights_indices, [params.size()[0], 1, 1])
        weights = torch.gather(params, 2, weights_indices)

        weights = torch.nn.functional.softmax(weights, dim=1)

        alignment_probs = []
        for j in range(enc_z.size()[1]):
            f1 = self._logistic(j + 0.5, means, scales)
            f2 = self._logistic(j - 0.5, means, scales)
            alignment_probs += [ weights * (f1 - f2) ]
        
        alignment_probs = torch.cat(alignment_probs,dim=1)
        alignment_probs = torch.sum(alignment_probs, 2)

        weighted_enc_z = torch.unsqueeze(alignment_probs,2) * enc_z

        context = torch.unsqueeze(torch.sum(weighted_enc_z, dim=1), 1).to(device)

        return context, None, None
        
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


