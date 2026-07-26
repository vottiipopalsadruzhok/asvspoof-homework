import torch
from torch import nn

from src.model.mfm import MFM


class LCNN(nn.Module):
    """
    LightCNN countermeasure, following Table 1 of the Speech Technology
    Center paper (https://arxiv.org/abs/1904.05576).
    """

    def __init__(self, n_freq=257, n_time=750, n_class=2, dropout=0.75):
        """
        Args:
            n_freq (int): number of frequency bins of the input spectrogram
                (n_fft // 2 + 1).
            n_time (int): number of time frames of the input spectrogram.
            n_class (int): number of output classes.
            dropout (float): dropout probability in the classifier head.
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        n_flatten = self._flatten_size(n_freq, n_time)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 160),
            MFM(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(80),
            nn.Linear(80, n_class),
        )

    def _flatten_size(self, n_freq, n_time):
        """
        Compute the flattened feature size after the convolutional trunk by
        passing a dummy tensor through it.

        Args:
            n_freq (int): number of frequency bins.
            n_time (int): number of time frames.
        Returns:
            n_flatten (int): number of features after flattening.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_freq, n_time)
            return self.features(dummy).flatten(1).shape[1]

    def forward(self, spectrogram, **batch):
        """
        Args:
            spectrogram (Tensor): input spectrogram of shape (N, 1, F, T).
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.classifier(self.features(spectrogram))}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
