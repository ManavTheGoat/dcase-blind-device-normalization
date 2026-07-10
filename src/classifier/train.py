import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def train_epoch(model, loader, optimizer, criterion, device, use_signal_feats=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc='Train', leave=False):
        mel, scene_labels, device_labels, audio_np = batch
        mel = mel.to(device)
        scene_labels = scene_labels.to(device)

        signal_feats = None
        if use_signal_feats:
            from src.signal_features.extractor import extract_batch_features
            feats_np = extract_batch_features(audio_np.numpy())
            signal_feats = torch.from_numpy(feats_np).to(device)

        optimizer.zero_grad()
        logits = model(mel, signal_feats)
        loss = criterion(logits, scene_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mel.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == scene_labels).sum().item()
        total += mel.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, use_signal_feats=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval', leave=False):
            mel, scene_labels, device_labels, audio_np = batch
            mel = mel.to(device)
            scene_labels = scene_labels.to(device)

            signal_feats = None
            if use_signal_feats:
                from src.signal_features.extractor import extract_batch_features
                feats_np = extract_batch_features(audio_np.numpy())
                signal_feats = torch.from_numpy(feats_np).to(device)

            logits = model(mel, signal_feats)
            loss = criterion(logits, scene_labels)

            total_loss += loss.item() * mel.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == scene_labels).sum().item()
            total += mel.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(scene_labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)
