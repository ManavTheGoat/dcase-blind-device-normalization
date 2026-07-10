"""
Phase 1: Train DevNormNet on paired (Si, Device A) clips.

The network learns to map simulated-device spectrograms back to Device A space.
At inference time, it generalizes to real devices B and C (zero-shot).

Run from project root:
    conda run -n dcase python experiments/train_devnorm.py
"""

import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.normalization.devnorm import DevNormNet, PairedDataset

DATASET_ROOT = 'data/raw/TAU-urban-acoustic-scenes-2022-mobile-development'
META_CSV     = os.path.join(DATASET_ROOT, 'meta.csv')
AUDIO_DIR    = DATASET_ROOT

BATCH_SIZE = 64
EPOCHS     = 30
LR         = 1e-3
VAL_SPLIT  = 0.1
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Device: {DEVICE}')

# Data
full_ds = PairedDataset(META_CSV, AUDIO_DIR)
n_val   = int(len(full_ds) * VAL_SPLIT)
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                 generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f'Train pairs: {len(train_ds):,} | Val pairs: {len(val_ds):,}')

# Model
model = DevNormNet(channels=8).to(DEVICE)
print(f'DevNormNet parameters: {model.count_parameters():,}')

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.L1Loss()  # L1 on mel-dB scale

best_val_loss = float('inf')
log = []

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    for mel_si, mel_a, _ in tqdm(train_loader, desc=f'Epoch {epoch} Train', leave=False):
        mel_si, mel_a = mel_si.to(DEVICE), mel_a.to(DEVICE)
        optimizer.zero_grad()
        out = model(mel_si)
        loss = criterion(out, mel_a)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * mel_si.size(0)
    train_loss /= len(train_ds)

    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for mel_si, mel_a, _ in val_loader:
            mel_si, mel_a = mel_si.to(DEVICE), mel_a.to(DEVICE)
            val_loss += criterion(model(mel_si), mel_a).item() * mel_si.size(0)
    val_loss /= len(val_ds)
    scheduler.step()

    print(f'Epoch {epoch:3d} | Train L1: {train_loss:.4f} | Val L1: {val_loss:.4f}')
    log.append({'epoch': epoch, 'train_loss': round(train_loss, 4),
                'val_loss': round(val_loss, 4)})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'results/devnorm_best.pt')

os.makedirs('results', exist_ok=True)
with open('results/devnorm_training_log.json', 'w') as f:
    json.dump(log, f, indent=2)

print(f'\nBest Val L1 Loss: {best_val_loss:.4f}')
print('Model saved to results/devnorm_best.pt')
