"""
Advanced experiments — train all three approaches and compare.

Run from project root:
    python experiments/train_advanced.py
"""

import os, sys, json, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.dataset import TAUDataset
from src.normalization.freqattn import FreqAttn
from src.normalization.adversarial import AdversarialASC
from src.classifier.specvit import SpecViT
from src.classifier.model import CPMobile

DATASET_ROOT = 'data/raw/TAU-urban-acoustic-scenes-2022-mobile-development'
CACHE_DIR    = 'data/mel_cache'
TRAIN_CSV    = os.path.join(DATASET_ROOT, 'evaluation_setup/fold1_train.csv')
VAL_CSV      = os.path.join(DATASET_ROOT, 'evaluation_setup/fold1_evaluate.csv')
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE   = 128
EPOCHS       = 50
SCENE_NAMES  = ['airport','bus','metro','metro_station','park',
                'public_square','shopping_mall','street_pedestrian','street_traffic','tram']

os.makedirs('results', exist_ok=True)


def get_loaders():
    train_ds = TAUDataset(TRAIN_CSV, DATASET_ROOT, cache_dir=CACHE_DIR)
    val_ds   = TAUDataset(VAL_CSV,   DATASET_ROOT, cache_dir=CACHE_DIR)
    kw = dict(batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, persistent_workers=True)
    return (DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw),
            len(train_ds), len(val_ds))


# ══════════════════════════════════════════════════════════════════════
# A — FreqAttn + CPMobile
# ══════════════════════════════════════════════════════════════════════
def run_freqattn():
    print("\n" + "═"*50)
    print(" EXPERIMENT A — FreqAttn + CPMobile")
    print("═"*50)

    train_loader, val_loader, n_train, n_val = get_loaders()

    freqattn  = FreqAttn(n_mels=128, d_model=32, n_heads=4, n_layers=2).to(DEVICE)
    classifier = CPMobile(n_classes=10).to(DEVICE)
    total = freqattn.count_parameters() + classifier.count_parameters()
    print(f"FreqAttn params: {freqattn.count_parameters():,}")
    print(f"CPMobile params: {classifier.count_parameters():,}")
    print(f"Total params:    {total:,} / 128,000 limit")

    # Train FreqAttn first (phase 1: mel reconstruction on all devices vs A)
    from src.normalization.devnorm import PairedDataset
    from torch.utils.data import random_split
    META_CSV = os.path.join(DATASET_ROOT, 'meta.csv')
    full_ds  = PairedDataset(META_CSV, DATASET_ROOT, cache_dir=CACHE_DIR)
    n_v      = int(len(full_ds) * 0.1)
    tr_ds, vl_ds = random_split(full_ds, [len(full_ds)-n_v, n_v],
                                  generator=torch.Generator().manual_seed(42))
    tr_l = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=8, pin_memory=True, persistent_workers=True)
    vl_l = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=8, pin_memory=True, persistent_workers=True)

    opt_fa = torch.optim.Adam(freqattn.parameters(), lr=1e-3)
    sch_fa = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fa, T_max=20)
    l1     = nn.L1Loss()
    best_l1 = float('inf')
    print("\n[Phase 1] Training FreqAttn normalizer ...")
    for epoch in range(1, 21):
        freqattn.train(); tl = 0.0
        for mel_si, mel_a, _ in tr_l:
            mel_si, mel_a = mel_si.to(DEVICE), mel_a.to(DEVICE)
            opt_fa.zero_grad()
            loss = l1(freqattn(mel_si), mel_a)
            loss.backward(); opt_fa.step()
            tl += loss.item() * mel_si.size(0)
        tl /= len(tr_ds)
        freqattn.eval(); vl_loss = 0.0
        with torch.no_grad():
            for mel_si, mel_a, _ in vl_l:
                vl_loss += l1(freqattn(mel_si.to(DEVICE)), mel_a.to(DEVICE)).item() * mel_si.size(0)
        vl_loss /= len(vl_ds); sch_fa.step()
        print(f"  Epoch {epoch:2d} | Train L1: {tl:.4f} | Val L1: {vl_loss:.4f}", flush=True)
        if vl_loss < best_l1:
            best_l1 = vl_loss
            torch.save(freqattn.state_dict(), 'results/freqattn_best.pt')

    freqattn.load_state_dict(torch.load('results/freqattn_best.pt'))
    for p in freqattn.parameters(): p.requires_grad = False
    freqattn.eval()
    print(f"FreqAttn frozen. Best L1: {best_l1:.4f}")

    # Phase 2: Train CPMobile on FreqAttn-normalized mels
    print("\n[Phase 2] Training CPMobile on FreqAttn output ...")
    opt_cls = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    sch_cls = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cls, T_max=EPOCHS)
    ce      = nn.CrossEntropyLoss()
    best_acc, log = 0.0, []
    all_preds, all_labels = [], []

    for epoch in range(1, EPOCHS+1):
        classifier.train(); correct, total = 0, 0
        for mel, scene_lbl, _, _ in train_loader:
            mel, scene_lbl = mel.to(DEVICE), scene_lbl.to(DEVICE)
            with torch.no_grad(): mel = freqattn(mel)
            opt_cls.zero_grad()
            logits = classifier(mel)
            ce(logits, scene_lbl).backward(); opt_cls.step()
            correct += (logits.argmax(1) == scene_lbl).sum().item(); total += mel.size(0)
        ta = correct / total

        classifier.eval(); vc, vt = 0, 0; all_preds, all_labels = [], []
        with torch.no_grad():
            for mel, scene_lbl, _, _ in val_loader:
                mel, scene_lbl = mel.to(DEVICE), scene_lbl.to(DEVICE)
                preds = classifier(freqattn(mel)).argmax(1)
                vc += (preds == scene_lbl).sum().item(); vt += mel.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(scene_lbl.cpu().numpy())
        va = vc / vt; sch_cls.step()
        print(f"Epoch {epoch:3d} | Train {ta*100:.1f}% | Val {va*100:.1f}%", flush=True)
        log.append({'epoch': epoch, 'train_acc': round(ta*100,2), 'val_acc': round(va*100,2)})
        if va > best_acc:
            best_acc = va
            torch.save(classifier.state_dict(), 'results/freqattn_classifier.pt')

    print(f"\n[A] Best Val Acc: {best_acc*100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=SCENE_NAMES))
    with open('results/freqattn_log.json', 'w') as f: json.dump(log, f, indent=2)
    return best_acc


# ══════════════════════════════════════════════════════════════════════
# B — Adversarial Domain Adaptation
# ══════════════════════════════════════════════════════════════════════
def run_adversarial():
    print("\n" + "═"*50)
    print(" EXPERIMENT B — Adversarial Domain Adaptation")
    print("═"*50)

    train_loader, val_loader, n_train, n_val = get_loaders()

    model = AdversarialASC(n_classes=10, n_devices=9).to(DEVICE)
    print(f"Total params: {model.count_parameters():,} / 128,000 limit")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()

    best_acc, log = 0.0, []
    all_preds, all_labels = [], []

    for epoch in range(1, EPOCHS+1):
        # Linearly ramp alpha 0→1 over first 10 epochs, then hold at 1
        alpha = min(1.0, epoch / 10.0)
        model.train(); correct, total = 0, 0; scene_loss_sum = 0.0; dev_loss_sum = 0.0

        for mel, scene_lbl, device_lbl, _ in train_loader:
            mel        = mel.to(DEVICE)
            scene_lbl  = scene_lbl.to(DEVICE)
            device_lbl = device_lbl.to(DEVICE)
            optimizer.zero_grad()
            scene_logits, device_logits = model(mel, alpha=alpha)
            loss_scene  = ce(scene_logits, scene_lbl)
            loss_device = ce(device_logits, device_lbl)
            loss = loss_scene + 0.1 * loss_device  # weight device loss lower
            loss.backward(); optimizer.step()
            correct += (scene_logits.argmax(1) == scene_lbl).sum().item()
            total   += mel.size(0)
            scene_loss_sum += loss_scene.item(); dev_loss_sum += loss_device.item()
        ta = correct / total

        model.eval(); vc, vt = 0, 0; all_preds, all_labels = [], []
        with torch.no_grad():
            for mel, scene_lbl, _, _ in val_loader:
                mel, scene_lbl = mel.to(DEVICE), scene_lbl.to(DEVICE)
                preds = model.predict(mel).argmax(1)
                vc += (preds == scene_lbl).sum().item(); vt += mel.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(scene_lbl.cpu().numpy())
        va = vc / vt; scheduler.step()
        print(f"Epoch {epoch:3d} | Train {ta*100:.1f}% | Val {va*100:.1f}% | α={alpha:.2f}", flush=True)
        log.append({'epoch': epoch, 'train_acc': round(ta*100,2), 'val_acc': round(va*100,2)})
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), 'results/adversarial_best.pt')

    print(f"\n[B] Best Val Acc: {best_acc*100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=SCENE_NAMES))
    with open('results/adversarial_log.json', 'w') as f: json.dump(log, f, indent=2)
    return best_acc


# ══════════════════════════════════════════════════════════════════════
# C — SpecViT Patch Transformer
# ══════════════════════════════════════════════════════════════════════
def run_specvit():
    print("\n" + "═"*50)
    print(" EXPERIMENT C — SpecViT Patch Transformer")
    print("═"*50)

    train_loader, val_loader, n_train, n_val = get_loaders()

    model = SpecViT(n_classes=10, patch_h=16, patch_w=16,
                    d_model=64, n_heads=4, n_layers=3).to(DEVICE)
    print(f"Total params: {model.count_parameters():,} / 128,000 limit")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()

    best_acc, log = 0.0, []
    all_preds, all_labels = [], []

    for epoch in range(1, EPOCHS+1):
        model.train(); correct, total = 0, 0
        for mel, scene_lbl, _, _ in train_loader:
            mel, scene_lbl = mel.to(DEVICE), scene_lbl.to(DEVICE)
            optimizer.zero_grad()
            logits = model(mel)
            ce(logits, scene_lbl).backward(); optimizer.step()
            correct += (logits.argmax(1) == scene_lbl).sum().item(); total += mel.size(0)
        ta = correct / total

        model.eval(); vc, vt = 0, 0; all_preds, all_labels = [], []
        with torch.no_grad():
            for mel, scene_lbl, _, _ in val_loader:
                mel, scene_lbl = mel.to(DEVICE), scene_lbl.to(DEVICE)
                preds = model(mel).argmax(1)
                vc += (preds == scene_lbl).sum().item(); vt += mel.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(scene_lbl.cpu().numpy())
        va = vc / vt; scheduler.step()
        print(f"Epoch {epoch:3d} | Train {ta*100:.1f}% | Val {va*100:.1f}%", flush=True)
        log.append({'epoch': epoch, 'train_acc': round(ta*100,2), 'val_acc': round(va*100,2)})
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), 'results/specvit_best.pt')

    print(f"\n[C] Best Val Acc: {best_acc*100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=SCENE_NAMES))
    with open('results/specvit_log.json', 'w') as f: json.dump(log, f, indent=2)
    return best_acc


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    import torch; print(f"GPU: {torch.cuda.get_device_name(0)}")

    acc_a = run_freqattn()
    acc_b = run_adversarial()
    acc_c = run_specvit()

    print("\n" + "═"*50)
    print(" FINAL RESULTS SUMMARY")
    print("═"*50)
    print(f"  DCASE 2025 baseline:         50.72%")
    print(f"  Baseline CPMobile (prev):    43.59%")
    print(f"  DevNormNet + CPMobile:       44.92%")
    print(f"  [A] FreqAttn + CPMobile:     {acc_a*100:.2f}%")
    print(f"  [B] Adversarial DA:          {acc_b*100:.2f}%")
    print(f"  [C] SpecViT Transformer:     {acc_c*100:.2f}%")
