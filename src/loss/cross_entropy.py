import torch
from torch import nn


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy with per-class weights, which compensate for the roughly
    1:9 bonafide/spoof imbalance of the LA train partition.
    """

    def __init__(self, weight=None):
        """
        Args:
            weight (list[float] | None): per-class weights, ordered by label
                index (0: spoof, 1: bonafide). None disables the weighting.
        """
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing the 'loss' key.
        """
        return {"loss": self.loss(logits, labels)}
