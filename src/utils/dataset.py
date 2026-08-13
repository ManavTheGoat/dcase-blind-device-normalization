import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

SCENE_CLASSES = [
    'airport', 'bus', 'metro', 'metro_station', 'park',
    'public_square', 'shopping_mall', 'street_pedestrian',
    'street_traffic', 'tram'
]
DEVICE_CLASSES = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']


class TAUDataset(Dataset):
    def __init__(self, meta_csv, audio_dir, cache_dir=None, sr=22050, n_mels=128, duration=1.0):
        self.df = pd.read_csv(meta_csv, sep='\t')
        self.audio_dir = audio_dir
        self.cache_dir = cache_dir
        self.sr = sr
        self.n_mels = n_mels
        self.n_samples = int(sr * duration)
        self.scene2idx = {s: i for i, s in enumerate(SCENE_CLASSES)}
        self.device2idx = {d: i for i, d in enumerate(DEVICE_CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.cache_dir is not None:
            rel = row['filename'].replace('audio/', '').replace('.wav', '')
            npy_path = os.path.join(self.cache_dir, rel + '.npy')
            mel_db = np.load(npy_path)
            mel_tensor = torch.from_numpy(mel_db).unsqueeze(0)
            audio = np.zeros(self.n_samples, dtype=np.float32)
        else:
            filepath = os.path.join(self.audio_dir, row['filename'])
            audio, _ = librosa.load(filepath, sr=self.sr, mono=True, duration=1.0)
            if len(audio) < self.n_samples:
                audio = np.pad(audio, (0, self.n_samples - len(audio)))
            else:
                audio = audio[:self.n_samples]
            mel = librosa.feature.melspectrogram(
                y=audio, sr=self.sr, n_mels=self.n_mels, fmax=self.sr // 2
            )
            mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
            mel_tensor = torch.from_numpy(mel_db).unsqueeze(0)

        scene_label  = self.scene2idx[row['scene_label']]
        device       = row['filename'].split('-')[-1].replace('.wav', '').lower()
        device_label = self.device2idx.get(device, 0)
        return mel_tensor, scene_label, device_label, audio
