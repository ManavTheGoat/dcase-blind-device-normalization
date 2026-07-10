"""
Experiment 1: Reproduce the DCASE 2025 baseline result (~50.72% accuracy).
Run from project root:
    conda run -n dcase python experiments/run_baseline.py
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.dataset import TAUDataset
from src.classifier.model import CPMobile
from src.classifier.train import train_epoch, evaluate

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = 'data/raw/TAU-urban-acoustic-scenes-2022-mobile-development'
TRAIN_CSV    = os.path.join(DATASET_ROOT, 'evaluation_setup/fold1_train.csv')
VAL_CSV      = os.path.join(DATASET_ROOT, 'evaluation_setup/fold1_evaluate.csv')
AUDIO_DIR    = os.path.join(DATASET_ROOT, 'audio')

BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

USE_SIGNAL_FEATS = False   # ← Set to True to run the novel experiment

print(f'Device: {DEVICE}')
print(f'Using signal features: {USE_SIGNAL_FEATS}')

# ── Data ───────────────────────────────────────────────────────────────────────
train_ds = TAUDataset(TRAIN_CSV, AUDIO_DIR)
val_ds   = TAUDataset(VAL_CSV,   AUDIO_DIR)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

# ── Model ──────────────────────────────────────────────────────────────────────
signal_feat_dim = 5 if USE_SIGNAL_FEATS else 0
model = CPMobile(n_classes=10, signal_feat_dim=signal_feat_dim).to(DEVICE)
print(f'Parameters: {model.count_parameters():,} / 128,000 limit')
assert model.count_parameters() <= 128_000, 'Model exceeds DCASE parameter limit!'

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ──────────────────────────────────────────────────────────────
best_acc = 0.0
results_log = []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, DEVICE, USE_SIGNAL_FEATS
    )
    val_loss, val_acc, preds, labels = evaluate(
        model, val_loader, criterion, DEVICE, USE_SIGNAL_FEATS
    )
    scheduler.step()

    log = {
        'epoch': epoch,
        'train_loss': round(train_loss, 4),
        'train_acc': round(train_acc * 100, 2),
        'val_loss': round(val_loss, 4),
        'val_acc': round(val_acc * 100, 2),
    }
    results_log.append(log)
    print(f"Epoch {epoch:3d} | Train {train_acc*100:.1f}% | Val {val_acc*100:.1f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        tag = 'with_signal_feats' if USE_SIGNAL_FEATS else 'baseline'
        torch.save(model.state_dict(), f'results/best_model_{tag}.pt')

print(f'\nBest Val Accuracy: {best_acc * 100:.2f}%')
print(f'Target (DCASE 2025 baseline): 50.72%')

# Save results log
import json
tag = 'with_signal_feats' if USE_SIGNAL_FEATS else 'baseline'
with open(f'results/training_log_{tag}.json', 'w') as f:
    json.dump(results_log, f, indent=2)
print(f'Results saved to results/training_log_{tag}.json')
