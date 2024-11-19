import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np
import torchaudio

class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
        device='cpu'
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length, device=device).float()
        # window = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

    def forward(self, audioin):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect").squeeze(1)
        # fft = torch.stft(
        #     audio,
        #     n_fft=self.n_fft,
        #     hop_length=self.hop_length,
        #     win_length=self.win_length,
        #     window=self.window,
        #     center=False,
        #     return_complex = True
        # )
        # fft = torch.abs(fft)
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sampling_rate, 
                                                         n_fft=self.n_fft, 
                                                         hop_length=self.hop_length,
                                                         win_length=self.win_length,
                                                         n_mels = self.n_mel_channels,
                                                         f_min = self.mel_fmin,
                                                         f_max = self.mel_fmax,
                                                        #  window_fn=self.window,
                                                         center=False)
        mel_output_new = transform(audio)
        # mel_output = torch.matmul(self.mel_basis, torch.sum(torch.pow(fft, 2), dim=[-1]))
        # print(fft.shape)
        # mel_output = torch.matmul(self.mel_basis, torch.sum(torch.pow(fft, 2), dim=[-1]).squeeze())
        log_mel_spec = torch.log10(torch.clamp(mel_output_new, min=1e-5))
        return log_mel_spec