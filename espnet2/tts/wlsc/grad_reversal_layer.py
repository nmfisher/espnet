import torch

class GradientReversalLayer(torch.nn.Module):
    """Gradient reversal layer"""
    def __init__(
        self,
        beta=1
    ):
        """Initialize SC module.

        Args:
            idim (int) Dimension of input features (usually num_mels)
            nspeakers (int) number of speakers.
            hidden_size (int, optional) Dimension of FFN hidden size
        """
        super(GradientReversalLayer, self).__init__()
        # store the hyperparameters
        self.beta = beta
    
    def backward(self,grad_output):
        return self.beta * grad_output.neg()

    def forward(self, xs):
        return xs
