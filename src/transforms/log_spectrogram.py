import torch
import torchaudio
from torch import nn


class LogSpectrogram(nn.Module):
    """
    Computes a single-channel log-power spectrogram from a raw waveform.

    Front-end configuration follows the recipe described in
    https://arxiv.org/abs/2103.11326 (20 ms window, 10 ms shift, 512-point
    FFT), used as an instance transform on top of ASVSpoofDataset.
    """

    def __init__(
        self, sample_rate=16000, win_length_ms=20, hop_length_ms=10, n_fft=512, eps=1e-8
    ):
        """
        Args:
            sample_rate (int): sample rate of the input waveform, in Hz.
            win_length_ms (float): STFT window length, in milliseconds.
            hop_length_ms (float): STFT hop length, in milliseconds.
            n_fft (int): FFT size.
            eps (float): additive constant to avoid log(0).
        """
        super().__init__()

        self.eps = eps
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=int(sample_rate * win_length_ms / 1000),
            hop_length=int(sample_rate * hop_length_ms / 1000),
            power=2.0,
        )

    def forward(self, waveform):
        """
        Args:
            waveform (Tensor): raw waveform of shape (1, T).
        Returns:
            log_spectrogram (Tensor): log-power spectrogram of shape
                (1, n_fft // 2 + 1, num_frames).
        """
        return torch.log(self.spectrogram(waveform) + self.eps)
