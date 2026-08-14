"""
Device-Conditioned Knowledge Distillation (DC-KD)

Novel contribution: different distillation temperature and alpha per device group.
The CP-JKU teacher ensemble was trained on Device A audio — its soft labels are
sharpest for Device A and softest for unseen simulated devices.
We exploit this by using higher temperature for consumer/simulated devices,
letting the student learn from softer (more uncertain) teacher distributions
where appropriate, and sharper distributions where the teacher is confident.

Architecture: FreqAttn (frozen, pre-trained) → CPMobile (trained with DC-KD)
Teacher: CP-JKU ensemble logits (6-model: 3×PaSST + 3×CP-ResNet)

Run from project root:
    python experiments/train_kd.py
"""

import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.dataset import TAUDataset
from src.normalization.freqattn import FreqAttn
from src.classifier.model import CPMobileWide

DATASET_ROOT = 'data/raw/TAU-urban-acoustic-scenes-2022-mobile-development'
CACHE_DIR    = 'data/mel_cache'
TRAIN_CSV    = os.path.join(DATASET_ROOT, 'evaluation_setup/fold1_train.csv')
VAL_CSV      = os.path.join(DATASET_ROOT, 'evaluation_setup/fold1_evaluate.csv')
LOGITS_PATH  = 'resources/ensemble_logits.pt'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE   = 256
EPOCHS       = 100
N_TOTAL      = 230350  # half the logit tensor; second half = time-rolled versions

SCENE_NAMES = ['airport','bus','metro','metro_station','park',
               'public_square','shopping_mall','street_pedestrian','street_traffic','tram']

# Device-conditional KD hyperparameters (the novel contribution)
# T: distillation temperature — higher = softer, more uncertain teacher distribution
# alpha: weight on KD loss (1-alpha = weight on hard CE loss)
DC_KD_PARAMS = {
    'a':              {'T': 2.0, 'alpha': 0.70},  # professional: teacher very confident
    'consumer':       {'T': 3.0, 'alpha': 0.85},  # B/C: teacher less certain
    'simulated_seen': {'T': 2.5, 'alpha': 0.80},  # S1-S3: seen during teacher training
    'simulated_unseen': {'T': 3.5, 'alpha': 0.90},# S4-S6: hardest, most uncertain teacher
}

os.makedirs('results', exist_ok=True)


class IndexedTAUDataset(Dataset):
    """Wraps TAUDataset to also return the sample index for teacher logit lookup."""
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        mel, scene, device, audio = self.base[idx]
        return mel, scene, device, idx


def device_group(device_lbl_tensor):
    """Map device label tensor → group string list."""
    # DEVICE_CLASSES = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
    groups = []
    for d in device_lbl_tensor.tolist():
        if d == 0:
            groups.append('a')
        elif d in (1, 2):
            groups.append('consumer')
        elif d in (3, 4, 5):
            groups.append('simulated_seen')
        else:
            groups.append('simulated_unseen')
    return groups


def dc_kd_loss(student_logits, teacher_logits, device_lbls, hard_labels, ce_fn):
    """
    Device-Conditioned KD loss.
    Different temperature T and distillation weight alpha per device group.
    """
    total_loss = torch.tensor(0.0, device=student_logits.device)
    groups = device_group(device_lbls)
    teacher_logits = teacher_logits.to(student_logits.device).float()

    for group, params in DC_KD_PARAMS.items():
        mask = torch.tensor([g == group for g in groups], dtype=torch.bool)
        if mask.sum() == 0:
            continue
        T, alpha = params['T'], params['alpha']
        s = student_logits[mask]
        t = teacher_logits[mask]
        h = hard_labels[mask]

        # KL divergence scaled by T^2 (standard KD formulation)
        kl = F.kl_div(
            F.log_softmax(s / T, dim=1),
            F.softmax(t / T, dim=1),
            reduction='batchmean'
        ) * (T ** 2)

        ce_loss = ce_fn(s, h)
        total_loss = total_loss + alpha * kl + (1.0 - alpha) * ce_loss

    return total_loss / len(DC_KD_PARAMS)





def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 60)
    print(" Device-Conditioned Knowledge Distillation (DC-KD)")
    print(" FreqAttn (frozen) -> CPMobileWide(bc=40) + DC-KD + TimeRoll")
    print("=" * 60)

    # Load teacher logits into memory (8.8MB — trivial)
    teacher_logits_all = torch.load(LOGITS_PATH, map_location='cpu').float()
    print(f"Teacher logits loaded: {teacher_logits_all.shape}  (float16→float32)")

    # Load pre-trained FreqAttn
    freqattn = FreqAttn(n_mels=128, d_model=32, n_heads=4, n_layers=2).to(DEVICE)
    if os.path.exists('results/best_freqattn.pt'):
        freqattn.load_state_dict(torch.load('results/best_freqattn.pt', map_location=DEVICE))
        print("FreqAttn loaded from results/best_freqattn.pt")
    else:
        print("WARNING: best_freqattn.pt not found — using random FreqAttn weights")
    for p in freqattn.parameters():
        p.requires_grad = False
    freqattn.eval()

    classifier = CPMobileWide(n_classes=10, bc=40).to(DEVICE)
    total = freqattn.count_parameters() + classifier.count_parameters()
    print(f"FreqAttn params: {freqattn.count_parameters():,}  (frozen)")
    print(f"CPMobileWide params: {classifier.count_parameters():,}  (trainable)")
    print(f"Total:           {total:,} / 128,000 limit")

    # Datasets
    train_base = TAUDataset(TRAIN_CSV, DATASET_ROOT, cache_dir=CACHE_DIR)
    val_base   = TAUDataset(VAL_CSV,   DATASET_ROOT, cache_dir=CACHE_DIR)
    train_ds   = IndexedTAUDataset(train_base)
    val_ds     = IndexedTAUDataset(val_base)

    kw = dict(batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ce_fn     = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, log = 0.0, []
    all_preds, all_labels = [], []

    print(f"\nDC-KD params by device group:")
    for g, p in DC_KD_PARAMS.items():
        print(f"  {g:20s}: T={p['T']}, alpha={p['alpha']}")
    print(f"\nTraining {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        classifier.train()
        correct, total_n, kd_loss_sum = 0, 0, 0.0

        for mel, scene_lbl, device_lbl, idx in train_loader:
            mel        = mel.to(DEVICE)
            scene_lbl  = scene_lbl.to(DEVICE)
            device_lbl = device_lbl.to(DEVICE)
            # Vectorized time-roll augmentation with matched teacher logits
            B, C, F, Tm = mel.shape
            roll_mask = torch.rand(B) < 0.5          # which samples to roll
            shifts    = torch.randint(1, Tm, (B,))   # random shift per sample
            # Teacher logits: use second half for rolled samples
            t_idx    = torch.where(roll_mask, idx + N_TOTAL, idx)
            t_logits = teacher_logits_all[t_idx]
            # Apply circular time shift sample-by-sample (torch.roll is fast on small T)
            for i in range(B):
                if roll_mask[i]:
                    mel[i] = torch.roll(mel[i], shifts=shifts[i].item(), dims=-1)

            with torch.no_grad():
                mel = freqattn(mel)

            optimizer.zero_grad()
            logits = classifier(mel)
            loss   = dc_kd_loss(logits, t_logits, device_lbl, scene_lbl, ce_fn)
            loss.backward()
            optimizer.step()

            correct  += (logits.argmax(1) == scene_lbl).sum().item()
            total_n  += mel.size(0)
            kd_loss_sum += loss.item()

        ta = correct / total_n

        classifier.eval()
        vc, vt = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mel, scene_lbl, _, _ in val_loader:
                mel, scene_lbl = mel.to(DEVICE), scene_lbl.to(DEVICE)
                preds = classifier(freqattn(mel)).argmax(1)
                vc += (preds == scene_lbl).sum().item(); vt += mel.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(scene_lbl.cpu().numpy())
        va = vc / vt
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | Train {ta*100:.1f}% | Val {va*100:.1f}% "
              f"| KD loss {kd_loss_sum/len(train_loader):.4f}", flush=True)
        log.append({'epoch': epoch, 'train_acc': round(ta*100,2), 'val_acc': round(va*100,2)})
        if va > best_acc:
            best_acc = va
            torch.save(classifier.state_dict(), 'results/kd_wide_roll_best_classifier.pt')

    print(f"\n{'='*60}")
    print(f"  DCASE 2025 baseline:        50.72%")
    print(f"  FreqAttn (no KD):           45.23%")
    print(f"  FreqAttn + DC-KD + TimeRoll:{best_acc*100:.2f}%")
    delta = best_acc*100 - 50.72
    print(f"  vs DCASE baseline:          {delta:+.2f}pp")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=SCENE_NAMES))
    with open('results/kd_wide_roll_log.json', 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == '__main__':
    main()
