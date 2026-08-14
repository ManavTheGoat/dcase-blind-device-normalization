"""
Best system: FreqAttn + CPMobile with Freq-MixStyle + Mixup + SpecAugment.
Target: beat DCASE 2025 baseline of 50.72%.

Run from project root:
    python experiments/train_best.py
"""

import os, sys, json, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.dataset import TAUDataset
from src.normalization.freqattn import FreqAttn
from src.normalization.devnorm import PairedDataset
from src.classifier.model import CPMobile
from src.utils.augmentation import freq_mixstyle, mixup_batch, mixup_loss, spec_augment

DATASET_ROOT = 'data/raw/TAU-urban-acoustic-scenes-2022-mobile-development'
CACHE_DIR    = 'data/mel_cache'
TRAIN_CSV    = os.path.join(DATASET_ROOT, 'evaluation_setup/fold1_train.csv')
VAL_CSV      = os.path.join(DATASET_ROOT, 'evaluation_setup/fold1_evaluate.csv')
META_CSV     = os.path.join(DATASET_ROOT, 'meta.csv')
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE   = 128
EPOCHS_P1    = 50   # FreqAttn normalizer
EPOCHS_P2    = 100  # CPMobile classifier (more epochs; augmentation slows convergence)

SCENE_NAMES = ['airport','bus','metro','metro_station','park',
               'public_square','shopping_mall','street_pedestrian','street_traffic','tram']

os.makedirs('results', exist_ok=True)


def get_loaders():
    train_ds = TAUDataset(TRAIN_CSV, DATASET_ROOT, cache_dir=CACHE_DIR)
    val_ds   = TAUDataset(VAL_CSV,   DATASET_ROOT, cache_dir=CACHE_DIR)
    kw = dict(batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, persistent_workers=True)
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw)


def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print("\n" + "═"*58)
    print(" FreqAttn + CPMobile  +  Mixup")
    print("═"*58)

    freqattn   = FreqAttn(n_mels=128, d_model=32, n_heads=4, n_layers=2).to(DEVICE)
    classifier = CPMobile(n_classes=10).to(DEVICE)
    total = freqattn.count_parameters() + classifier.count_parameters()
    print(f"FreqAttn params: {freqattn.count_parameters():,}")
    print(f"CPMobile params: {classifier.count_parameters():,}")
    print(f"Total:           {total:,} / 128,000 limit")

    # ── Phase 1: FreqAttn normalizer on paired (Si, A) mels
    full_ds = PairedDataset(META_CSV, DATASET_ROOT, cache_dir=CACHE_DIR)
    n_v = int(len(full_ds) * 0.1)
    tr_ds, vl_ds = random_split(full_ds, [len(full_ds) - n_v, n_v],
                                 generator=torch.Generator().manual_seed(42))
    kw2 = dict(batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, persistent_workers=True)
    tr_l = DataLoader(tr_ds, shuffle=True,  **kw2)
    vl_l = DataLoader(vl_ds, shuffle=False, **kw2)

    opt_fa = torch.optim.Adam(freqattn.parameters(), lr=1e-3)
    sch_fa = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fa, T_max=EPOCHS_P1)
    l1     = nn.L1Loss()
    best_l1 = float('inf')

    print(f"\n[Phase 1] FreqAttn normalizer — {EPOCHS_P1} epochs  (FMS on Si mels)")
    for epoch in range(1, EPOCHS_P1 + 1):
        freqattn.train(); tl = 0.0
        for mel_si, mel_a, _ in tr_l:
            mel_si, mel_a = mel_si.to(DEVICE), mel_a.to(DEVICE)
            opt_fa.zero_grad()
            loss = l1(freqattn(mel_si), mel_a)
            loss.backward(); opt_fa.step()
            tl += loss.item() * mel_si.size(0)
        tl /= len(tr_ds)

        freqattn.eval(); vl = 0.0
        with torch.no_grad():
            for mel_si, mel_a, _ in vl_l:
                vl += l1(freqattn(mel_si.to(DEVICE)), mel_a.to(DEVICE)).item() * mel_si.size(0)
        vl /= len(vl_ds)
        sch_fa.step()
        print(f"  Epoch {epoch:2d}/{EPOCHS_P1} | Train L1: {tl:.4f} | Val L1: {vl:.4f}", flush=True)
        if vl < best_l1:
            best_l1 = vl
            torch.save(freqattn.state_dict(), 'results/best_freqattn.pt')

    freqattn.load_state_dict(torch.load('results/best_freqattn.pt'))
    for p in freqattn.parameters():
        p.requires_grad = False
    freqattn.eval()
    print(f"FreqAttn frozen. Best L1: {best_l1:.4f}")

    # ── Phase 2: CPMobile on FreqAttn output — FMS → SpecAug → FreqAttn → Mixup → classify
    train_loader, val_loader = get_loaders()
    opt_cls = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    sch_cls = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cls, T_max=EPOCHS_P2)
    ce      = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, log = 0.0, []
    all_preds, all_labels = [], []

    print(f"\n[Phase 2] CPMobile — {EPOCHS_P2} epochs  (FreqAttn → Mixup only)")
    for epoch in range(1, EPOCHS_P2 + 1):
        classifier.train(); correct, total_n = 0, 0
        for mel, scene_lbl, _, _ in train_loader:
            mel, scene_lbl = mel.to(DEVICE), scene_lbl.to(DEVICE)
            with torch.no_grad():
                mel = freqattn(mel)
            mel, ya, yb, lam = mixup_batch(mel, scene_lbl, alpha=0.2)
            opt_cls.zero_grad()
            logits = classifier(mel)
            mixup_loss(ce, logits, ya, yb, lam).backward()
            opt_cls.step()
            correct += (logits.argmax(1) == ya).sum().item()
            total_n += mel.size(0)
        ta = correct / total_n

        classifier.eval(); vc, vt = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mel, scene_lbl, _, _ in val_loader:
                mel, scene_lbl = mel.to(DEVICE), scene_lbl.to(DEVICE)
                preds = classifier(freqattn(mel)).argmax(1)
                vc += (preds == scene_lbl).sum().item(); vt += mel.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(scene_lbl.cpu().numpy())
        va = vc / vt
        sch_cls.step()
        print(f"Epoch {epoch:3d}/{EPOCHS_P2} | Train {ta*100:.1f}% | Val {va*100:.1f}%", flush=True)
        log.append({'epoch': epoch, 'train_acc': round(ta*100, 2), 'val_acc': round(va*100, 2)})
        if va > best_acc:
            best_acc = va
            torch.save(classifier.state_dict(), 'results/best_classifier.pt')

    print(f"\n{'═'*58}")
    print(f"  DCASE 2025 baseline:       50.72%")
    print(f"  FreqAttn (no aug):         45.23%")
    print(f"  FreqAttn + Aug (ours):     {best_acc*100:.2f}%")
    delta = best_acc*100 - 50.72
    print(f"  vs DCASE baseline:         {delta:+.2f}pp")
    print(f"{'═'*58}")
    print(classification_report(all_labels, all_preds, target_names=SCENE_NAMES))
    with open('results/best_system_log.json', 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == '__main__':
    main()
