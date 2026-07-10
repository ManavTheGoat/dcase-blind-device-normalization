"""
Phase 2: Full pipeline — DevNormNet + CPMobile classifier.

Loads pretrained DevNormNet, freezes it, trains classifier on normalized mels.
Compare against baseline (no normalization) and oracle (device label as input).

Run from project root:
    conda run -n dcase python experiments/run_full_pipeline.py
"""

import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

from src.normalization.devnorm import DevNormNet
from src.classifier.model import CPMobile
from src.utils.dataset import TAUDataset

DATASET_ROOT = 'data/raw/TAU-urban-acoustic-scenes-2022-mobile-development'
TRAIN_CSV    = os.path.join(DATASET_ROOT, 'evaluation_setup/fold1_train.csv')
VAL_CSV      = os.path.join(DATASET_ROOT, 'evaluation_setup/fold1_evaluate.csv')
AUDIO_DIR    = DATASET_ROOT

BATCH_SIZE = 64
EPOCHS     = 50
LR         = 1e-3
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

SCENE_NAMES = ['airport','bus','metro','metro_station','park',
               'public_square','shopping_mall','street_pedestrian',
               'street_traffic','tram']

print(f'Device: {DEVICE}')

# Load DevNormNet (pretrained)
devnorm = DevNormNet(channels=8).to(DEVICE)
if os.path.exists('results/devnorm_best.pt'):
    devnorm.load_state_dict(torch.load('results/devnorm_best.pt', map_location=DEVICE))
    print('Loaded pretrained DevNormNet')
    for p in devnorm.parameters():
        p.requires_grad = False
    devnorm.eval()
else:
    print('WARNING: devnorm_best.pt not found. Run train_devnorm.py first.')
    devnorm = None

# Data
train_ds = TAUDataset(TRAIN_CSV, AUDIO_DIR)
val_ds   = TAUDataset(VAL_CSV,   AUDIO_DIR)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
print(f'Train: {len(train_ds):,} | Val: {len(val_ds):,}')

# Classifier
classifier = CPMobile(n_classes=10, signal_feat_dim=0).to(DEVICE)
total_params = classifier.count_parameters() + (devnorm.count_parameters() if devnorm else 0)
print(f'Total parameters: {total_params:,} / 128,000 limit')

optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

best_acc = 0.0
log = []

for epoch in range(1, EPOCHS + 1):
    classifier.train()
    total_loss, correct, total = 0.0, 0, 0

    for mel, scene_labels, device_labels, _ in tqdm(train_loader,
                                                     desc=f'Epoch {epoch}', leave=False):
        mel = mel.to(DEVICE)
        scene_labels = scene_labels.to(DEVICE)

        # Apply normalization if devnorm is loaded
        if devnorm is not None:
            with torch.no_grad():
                mel = devnorm(mel)

        optimizer.zero_grad()
        logits = classifier(mel)
        loss = criterion(logits, scene_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mel.size(0)
        correct += (logits.argmax(1) == scene_labels).sum().item()
        total += mel.size(0)

    train_acc = correct / total

    # Validation
    classifier.eval()
    val_correct, val_total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, scene_labels, _, _ in val_loader:
            mel = mel.to(DEVICE)
            scene_labels = scene_labels.to(DEVICE)
            if devnorm is not None:
                mel = devnorm(mel)
            preds = classifier(mel).argmax(1)
            val_correct += (preds == scene_labels).sum().item()
            val_total += mel.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(scene_labels.cpu().numpy())

    val_acc = val_correct / val_total
    scheduler.step()

    print(f'Epoch {epoch:3d} | Train {train_acc*100:.1f}% | Val {val_acc*100:.1f}%')
    log.append({'epoch': epoch, 'train_acc': round(train_acc*100, 2),
                'val_acc': round(val_acc*100, 2)})

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(classifier.state_dict(), 'results/classifier_with_devnorm.pt')

print(f'\nBest Val Accuracy: {best_acc*100:.2f}%')
print('\nPer-class breakdown:')
print(classification_report(all_labels, all_preds, target_names=SCENE_NAMES))

with open('results/pipeline_training_log.json', 'w') as f:
    json.dump(log, f, indent=2)
print('Results saved.')
