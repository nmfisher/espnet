#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HQSS encoder related modules."""

import six

import torch
import torch.distributions as D

def encoder_init(m):
    """Initialize encoder parameters."""
    #torch.nn.init.xavier_uniform_(m, torch.nn.init.calculate_gain("relu"))p
    pass

class ResidualEncoder(torch.nn.Module):
    """Encoder module of Spectrogram prediction network.

    This is a module of encoder of Spectrogram prediction network in HQSS,
    which described in `High Quality Streaming Speech Synthesis with Low, Sentence-Length-Independent Latency`_. This is the encoder which converts either a sequence
    of characters or acoustic features into the sequence of hidden states.

    .. _`High Quality Streaming Speech Synthesis with Low, Sentence-Length-Independent Latency`:
       https://arxiv.org/pdf/2111.09052.pdf

    """

    def __init__(
        self,
        idim=80,
        odim=128,
        hdim=512,
        num_gaussians=5,
        device="cuda"
    ):
        """Initialize residual encoder module.

        Args:
            idim (int) Dimension of the inputs.
            num_gaussians: Number of Gaussian mixture components to use
        """
        super(ResidualEncoder, self).__init__()
        self.odim = odim
        self.rnn = torch.nn.GRU(idim, hdim)
        self.proj = torch.nn.Linear(hdim, (num_gaussians*2*odim) + num_gaussians)
        self.num_gaussians = num_gaussians
        # self.weights = torch.Tensor(num_gaussians,requires_grad=True).to(device)
        # self.means = torch.Tensor((num_gaussians,odim), requires_grad=True).to(device)
        # self.stdevs = torch.Tensor((num_gaussians,odim),requires_grad=True).to(device)
        # mix = D.Categorical(self.weights)
        # self.comp = D.Independent(D.Normal(self.means,self.stdevs), 1)
        # self.gmm = D.MixtureSameFamily(mix, self.comp)

        self.apply(encoder_init)


    def forward(self, ys):
        """Samples from mixture components

        Args:
            ys (Tensor): Acoustic features (B, Tmax, idim * encoder_reduction_factor). Padded
                value should be 0.
            ilens (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Batch of the sequences of encoder states(B, Tmax, eunits).
            LongTensor: Batch of lengths of each sequence (B,)

        """
        device = ys.device
        
        batch_size = ys.size()[0]

        out,_ = self.rnn(ys)
        
        params = self.proj(out[:,-1,:])

        samples = torch.zeros(batch_size, 1, self.odim).to(ys.device)
        
        for j in range(batch_size):
          for i in range(self.num_gaussians):
            mu = params[j,i*self.odim:(i+1)*self.odim]
            var = params[j,2*i*self.odim:((i*2)+1)*self.odim] + 1e-6
            var_m = torch.tril(torch.diag(var))
            m = D.MultivariateNormal(mu, scale_tril=var_m, validate_args=False)
            s = m.sample()
            weight = params[j,i-self.num_gaussians]
          
            samples[j,0,:] += weight * s

        return samples

    def inference(self, durations, pitch):
        """Inference.

        Args:
            x (Tensor): The sequeunce of character ids (T,)
                    or acoustic feature (T, idim * encoder_reduction_factor).

        Returns:
            Tensor: The sequences of encoder states(T, eunits).

        """
        raise Exception()