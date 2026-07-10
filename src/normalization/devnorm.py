"""
DevNormNet — Learned Device Normalization Network

Learns to map any device's mel spectrogram toward Device A's acoustic space,
using paired (Device A, Simulated Device Si) clips as supervision.
At inference time it operates blindly — no device label required.
"""

import torch
import torch.nn as nn
import pandas as pd


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class DevNormNet(nn.Module):
    """
    Lightweight spectrogram normalization network (~8K params).

    Input:  (1, n_mels, T) mel spectrogram from any device
    Output: (1, n_mels, T) normalized spectrogram in Device A acoustic space

    Trained on paired (Si, A) clips where Si is a simulated device with known
    transfer function. At inference: applied blindly to real devices B and C.
    Network only learns the residual correction — input is always added back.
    """

    def __init__(self, channels=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock(channels)
        self.res2 = ResBlock(channels)
        self.decoder = nn.Conv2d(channels, 1, 3, padding=1, bias=False)

    def forward(self, x):
        h = self.encoder(x)
        h = self.res1(h)
        h = self.res2(h)
        return x + self.decoder(h)  # residual: only learn the correction

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PairedDataset(torch.utils.data.Dataset):
    """
    Paired (Device Si, Device A) mel spectrograms for DevNormNet training.
    For each simulated clip, finds its exact Device A counterpart.
    """

    def __init__(self, meta_csv, audio_dir, sr=22050, n_mels=128):
        import os
        meta = pd.read_csv(meta_csv, sep='\t')
        meta['device'] = meta['filename'].apply(
            lambda f: f.split('-')[-1].replace('.wav', '').lower()
        )

        def base_key(fname):
            stem = fname.replace('audio/', '').replace('.wav', '')
            return '-'.join(stem.split('-')[:-1])

        meta['base_key'] = meta['filename'].apply(base_key)
        device_a = meta[meta['device'] == 'a'].set_index('base_key')['filename'].to_dict()

        sim_meta = meta[meta['device'].isin(['s1','s2','s3','s4','s5','s6'])].copy()
        sim_meta = sim_meta[sim_meta['base_key'].isin(device_a)]
        sim_meta['filename_a'] = sim_meta['base_key'].map(device_a)

        self.pairs = list(zip(
            sim_meta['filename'].values,
            sim_meta['filename_a'].values,
            sim_meta['scene_label'].values
        ))
        self.audio_dir = audio_dir
        self.sr = sr
        self.n_mels = n_mels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        import librosa, numpy as np
        fname_si, fname_a, scene = self.pairs[idx]

        def load_mel(fname):
            path = self.audio_dir + '/' + fname
            audio, _ = librosa.load(path, sr=self.sr, mono=True, duration=1.0)
            if len(audio) < self.sr:
                audio = np.pad(audio, (0, self.sr - len(audio)))
            mel = librosa.feature.melspectrogram(y=audio, sr=self.sr,
                                                  n_mels=self.n_mels, fmax=self.sr//2)
            return librosa.power_to_db(mel, ref=np.max).astype('float32')[None]

        return torch.from_numpy(load_mel(fname_si)), torch.from_numpy(load_mel(fname_a)), scene
