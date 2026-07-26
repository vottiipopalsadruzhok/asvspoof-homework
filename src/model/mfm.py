import torch
from torch import nn


class MFM(nn.Module):
    """
    Max-Feature-Map activation (MFM 2/1).

    Splits the tensor in two halves along the channel axis and takes the
    elementwise maximum. The channel axis is 1 both for convolutional
    (N, C, H, W) and for fully-connected (N, F) features, so the same
    module serves both cases.
    """

    def forward(self, x):
        """
        Args:
            x (Tensor): tensor of shape (N, C, ...), C even.
        Returns:
            x (Tensor): tensor of shape (N, C // 2, ...).
        """
        a, b = torch.split(x, x.shape[1] // 2, dim=1)
        return torch.max(a, b)
