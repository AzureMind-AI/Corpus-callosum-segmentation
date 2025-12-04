# main.py
# 10-fold CV with 80/10 (of the 90%) train/val split inside each fold,
# pick best-vali-Dice model, evaluate on test, summarize metrics, and
# produce one aggregated train/val loss plot with std "cloud".

import os
import math
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib  # if you don't need niimg reading here, you can remove it
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt

# ---- local modules (keep file names as you uploaded) ----
from datasets import Dataset_MRI_2D_Fold  # uses your existing combined-fold dataset
from models import UNet, UNet_Enhanced     # your models module (typo kept intentionally)

# =========================
# Config
# =========================
CONFIG = {
    "LEARNING_RATE": 2e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE": 4,
    "NUM_EPOCHS": 30,  # fixed epochs per fold (no early stopping requested)
    "IMAGE_HEIGHT": 512,
    "IMAGE_WIDTH": 512,
    "IN_CHANNELS": 1,
    "NUM_CLASSES": 1,

    # Data dirs (same naming you used)
    "TRAIN_IMG_DIR": "data/train/images",
    "TRAIN_LBL_DIR": "data/train/labels",
    "TEST_IMG_DIR": "data/test/images",
    "TEST_LBL_DIR": "data/test/labels",

    "K_FOLDS": 10,
    "NUM_WORKERS": 4,
    "MODEL_NAME": "UNet",           # "UNet" or "UNet_Enhanced"
    "USE_MULTI_GPU": True,          # wraps with DataParallel if >1 GPUs
    "SAVE_DIR": "runs",             # where to put results
    "SEED": 42,

    # Deep supervision (only used by UNet_Enhanced when training)
    "USE_DEEP_SUPERVISION": False,
    "DEEP_SUPERVISION_WEIGHTS": [1.0, 0.8, 0.6, 0.4],
}

# =========================
# Utilities
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_multi_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available -> CPU.")
        return "cpu", 1, 1
    n = torch.cuda.device_count()
    if n > 1 and CONFIG["USE_MULTI_GPU"]:
        print(f"Using {n} GPUs (DataParallel).")
        return "cuda", n, n  # scale batch/lr by n
    print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    return "cuda", 1, 1

# =========================
# Metrics (binary segmentation)
# =========================
def _bin(preds):
    preds = torch.sigmoid(preds)
    return (preds > 0.5).float()

def dice_score(preds, targets, smooth=1e-6):
    preds = _bin(preds)
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return ((2.0 * inter + smooth) / (union + smooth)).item()

def jaccard_score(preds, targets, smooth=1e-6):
    preds = _bin(preds)
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()

def accuracy_score(preds, targets):
    preds = _bin(preds)
    correct = (preds == targets).sum()
    total = targets.numel()
    return (correct.float() / total).item()

def sensitivity_score(preds, targets, smooth=1e-6):
    preds = _bin(preds)
    tp = (preds * targets).sum()
    pos = targets.sum()
    return ((tp + smooth) / (pos + smooth)).item()

def specificity_score(preds, targets, smooth=1e-6):
    preds = _bin(preds)
    tn = ((1 - preds) * (1 - targets)).sum()
    neg = (1 - targets).sum()
    return ((tn + smooth) / (neg + smooth)).item()

def hausdorff_distance(preds, targets):
    try:
        preds = _bin(preds)
        p = preds.cpu().numpy().squeeze()
        t = targets.cpu().numpy().squeeze()
        pc = np.column_stack(np.where(p > 0))
        tc = np.column_stack(np.where(t > 0))
        if len(pc) == 0 and len(tc) == 0:
            return 0.0
        if len(pc) == 0 or len(tc) == 0:
            return float("inf")
        return max(directed_hausdorff(pc, tc)[0], directed_hausdorff(tc, pc)[0])
    except Exception:
        return float("inf")

def all_metrics(preds, targets) -> Dict[str, float]:
    return {
        "dice": dice_score(preds, targets),
        "jaccard": jaccard_score(preds, targets),
        "accuracy": accuracy_score(preds, targets),
        "sensitivity": sensitivity_score(preds, targets),
        "specificity": specificity_score(preds, targets),
        "hausdorff": hausdorff_distance(preds, targets),
    }

# =========================
# Train / Evaluate loops
# =========================
def train_one_epoch(model, loader, optimizer, loss_fn, device, use_deep_supervision=False, ds_weights=None):
    model.train()
    loop = tqdm(loader, leave=False, desc="train")
    total = 0.0

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        if use_deep_supervision and model.training:
            out = model(images, deep_supervision=True)
            if isinstance(out, tuple):
                main_pred, ds1, ds2, ds3 = out
                loss_main = loss_fn(main_pred, masks)
                loss_ds1 = loss_fn(ds1, masks)
                loss_ds2 = loss_fn(ds2, masks)
                loss_ds3 = loss_fn(ds3, masks)
                if ds_weights is None:
                    ds_weights = [1.0, 0.8, 0.6, 0.4]
                loss = (
                    ds_weights[0]*loss_main +
                    ds_weights[1]*loss_ds1 +
                    ds_weights[2]*loss_ds2 +
                    ds_weights[3]*loss_ds3
                )
            else:
                loss = loss_fn(out, masks)
        else:
            out = model(images)
            loss = loss_fn(out, masks)

        loss.backward()
        optimizer.step()
        total += loss.item()
        loop.set_postfix(loss=loss.item())

    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    loop = tqdm(loader, leave=False, desc="eval")
    tot_loss = 0.0
    agg = {k: [] for k in ["dice","jaccard","accuracy","sensitivity","specificity","hausdorff"]}

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        out = model(images)
        loss = loss_fn(out, masks)
        tot_loss += loss.item()

        m = all_metrics(out, masks)
        for k,v in m.items():
            if not (np.isnan(v) or np.isinf(v)):
                agg[k].append(v)

        loop.set_postfix(loss=loss.item(), dice=(agg["dice"][-1] if agg["dice"] else 0.0))

    avg = {k: (float(np.mean(v)) if len(v)>0 else 0.0) for k,v in agg.items()}
    return tot_loss / max(1, len(loader)), avg

# =========================
# Plot (single figure: avg curves + std cloud)
# =========================
def plot_avg_std(train_curves: List[List[float]], val_curves: List[List[float]], save_path: Path):
    """
    train_curves: list (fold) of lists (epoch) of train loss
    val_curves:   list (fold) of lists (epoch) of val loss
    Saves a single PNG with mean±std for both curves.
    """
    arr_train = np.array(train_curves, dtype=float)  # [F, E]
    arr_val   = np.array(val_curves, dtype=float)    # [F, E]

    mean_train = arr_train.mean(axis=0)
    std_train  = arr_train.std(axis=0)
    mean_val   = arr_val.mean(axis=0)
    std_val    = arr_val.std(axis=0)

    epochs = np.arange(1, mean_train.shape[0]+1)

    plt.figure(figsize=(9,6))
    # mean lines
    plt.plot(epochs, mean_train, label="Train loss")
    plt.plot(epochs, mean_val,   label="Val loss")

    # std "cloud"
    plt.fill_between(epochs, mean_train-std_train, mean_train+std_train, alpha=0.2)
    plt.fill_between(epochs, mean_val-std_val,     mean_val+std_val,     alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Average Train vs. Val Loss across 10 folds (±1 std)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# =========================
# Per-fold runner
# =========================
def run_one_fold(
    fold_idx: int,
    master: Dataset_MRI_2D_Fold,
    image_transform,
    mask_transform,
    device: str,
    gpu_mult: int,
    save_root: Path,
) -> Tuple[Dict[str, float], List[float], List[float]]:

    # Get fold datasets (90% train+val / 10% test provided by the KFold split)
    fold_train_ds, fold_test_ds = master.get_fold_datasets(fold_idx)

    # Inside 90%: split into 8/9 (train) and 1/9 (val) -> exactly 80/10 of total
    n_all = len(fold_train_ds)
    val_size = max(1, int(round(n_all / 9.0)))  # 1/9 of the 90% block
    train_size = n_all - val_size
    gen = torch.Generator().manual_seed(CONFIG["SEED"] + fold_idx)

    inner_train_ds, inner_val_ds = random_split(fold_train_ds, [train_size, val_size], generator=gen)

    # DataLoaders
    eff_batch = CONFIG["BATCH_SIZE"] * gpu_mult
    eff_workers = CONFIG["NUM_WORKERS"] * gpu_mult if device == "cuda" and gpu_mult > 1 else CONFIG["NUM_WORKERS"]

    train_loader = DataLoader(inner_train_ds, batch_size=eff_batch, shuffle=True,
                              num_workers=eff_workers, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(inner_val_ds,   batch_size=eff_batch, shuffle=False,
                              num_workers=eff_workers, pin_memory=(device=="cuda"))
    test_loader  = DataLoader(fold_test_ds,   batch_size=eff_batch, shuffle=False,
                              num_workers=eff_workers, pin_memory=(device=="cuda"))

    # Model
    if CONFIG["MODEL_NAME"].lower() == "unet_enhanced":
        model = UNet_Enhanced(in_channels=CONFIG["IN_CHANNELS"], n_classes=CONFIG["NUM_CLASSES"])
    else:
        model = UNet(in_channels=CONFIG["IN_CHANNELS"], n_classes=CONFIG["NUM_CLASSES"])

    if device == "cuda":
        model = model.cuda()
        if torch.cuda.device_count() > 1 and CONFIG["USE_MULTI_GPU"]:
            model = torch.nn.DataParallel(model)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"] * gpu_mult)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=False)

    # Train epochs, track best epoch by validation Dice
    best_val_dice = -1.0
    best_state = None
    train_curve, val_curve = [], []

    fold_dir = save_root / f"fold_{fold_idx+1:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(CONFIG["NUM_EPOCHS"]):
        print(f"\n[FOLD {fold_idx+1:02d}] Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']}")
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            use_deep_supervision=CONFIG["USE_DEEP_SUPERVISION"] and (CONFIG["MODEL_NAME"].lower()=="unet_enhanced"),
            ds_weights=CONFIG["DEEP_SUPERVISION_WEIGHTS"],
        )
        va_loss, va_metrics = evaluate(model, val_loader, loss_fn, device)

        train_curve.append(tr_loss)
        val_curve.append(va_loss)

        # Learning rate schedule on validation loss (common practice)
        scheduler.step(va_loss)

        # Keep best checkpoint by validation DICE
        cur_val_dice = va_metrics.get("dice", 0.0)
        if cur_val_dice > best_val_dice:
            best_val_dice = cur_val_dice
            best_state = (model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict())
            torch.save(best_state, fold_dir / "best_by_valdice.pth")

        # progress print
        print(f"  train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_dice={cur_val_dice:.4f}")

    # Load best and evaluate on TEST
    if best_state is not None:
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)

    test_loss, test_metrics = evaluate(model, test_loader, loss_fn, device)
    print(f"[FOLD {fold_idx+1:02d}] TEST -> loss={test_loss:.4f}, "
          f"dice={test_metrics['dice']:.4f}, jaccard={test_metrics['jaccard']:.4f}, "
          f"acc={test_metrics['accuracy']:.4f}, sens={test_metrics['sensitivity']:.4f}, "
          f"spec={test_metrics['specificity']:.4f}, hd={test_metrics['hausdorff']:.4f}")

    # Save per-fold test metrics
    with open(fold_dir / "test_metrics.json", "w") as f:
        json.dump({"test_loss": test_loss, **test_metrics, "best_val_dice": best_val_dice}, f, indent=2)

    # Return for aggregation
    return test_metrics, train_curve, val_curve

# =========================
# Main
# =========================
def main():
    set_seed(CONFIG["SEED"])
    device, n_gpus, gpu_mult = setup_multi_gpu()

    # IO setup
    save_root = Path(CONFIG["SAVE_DIR"])
    save_root.mkdir(parents=True, exist_ok=True)

    # Transforms (tensors in [C,H,W])
    image_transform = transforms.Compose([
        transforms.Resize((CONFIG["IMAGE_HEIGHT"], CONFIG["IMAGE_WIDTH"]), antialias=True),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((CONFIG["IMAGE_HEIGHT"], CONFIG["IMAGE_WIDTH"]), antialias=True),
    ])

    # Master dataset across both train/val/test dirs (your class handles KFold indices)
    print("Loading dataset for 10-fold CV...")
    master = Dataset_MRI_2D_Fold(
        train_image_dir=CONFIG["TRAIN_IMG_DIR"],
        train_label_dir=CONFIG["TRAIN_LBL_DIR"],
        test_image_dir=CONFIG["TEST_IMG_DIR"],
        test_label_dir=CONFIG["TEST_LBL_DIR"],
        image_transform=image_transform,
        mask_transform=mask_transform,
        k_folds=CONFIG["K_FOLDS"],
        verbose=True
    )
    print(f"Total samples: {len(master)}")

    # Storage for aggregation
    per_fold_metrics: List[Dict[str,float]] = []
    train_curves: List[List[float]] = []
    val_curves: List[List[float]] = []

    # Run folds
    for f in range(CONFIG["K_FOLDS"]):
        m, tr_c, va_c = run_one_fold(
            fold_idx=f,
            master=master,
            image_transform=image_transform,
            mask_transform=mask_transform,
            device=device,
            gpu_mult=gpu_mult,
            save_root=save_root
        )
        per_fold_metrics.append(m)
        train_curves.append(tr_c)
        val_curves.append(va_c)

    # =========================
    # Summaries
    # =========================
    metric_names = ["dice","jaccard","accuracy","sensitivity","specificity","hausdorff"]
    means = {}
    stds  = {}

    print("\n========== PER-FOLD TEST METRICS ==========")
    for i, m in enumerate(per_fold_metrics, start=1):
        print(f"Fold {i:02d}: " +
              ", ".join([f"{k}={m.get(k, 0.0):.4f}" for k in metric_names]))

    print("\n============= SUMMARY (TEST) =============")
    for k in metric_names:
        arr = np.array([m.get(k, 0.0) for m in per_fold_metrics], dtype=float)
        means[k] = float(np.mean(arr))
        stds[k]  = float(np.std(arr))
        print(f"{k:>11}: {means[k]:.4f} ± {stds[k]:.4f}")

    # Save CSV with per-fold metrics
    try:
        import pandas as pd
        rows = []
        for i, m in enumerate(per_fold_metrics, start=1):
            row = {"fold": i}
            row.update({k: m.get(k, 0.0) for k in metric_names})
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(save_root / "per_fold_test_metrics.csv", index=False)

        # Means/stds
        df_summary = pd.DataFrame([{"metric": k, "mean": means[k], "std": stds[k]} for k in metric_names])
        df_summary.to_csv(save_root / "test_metrics_summary.csv", index=False)
    except Exception as e:
        print("Pandas not available? Skipping CSV save.", e)

    # Save JSON summary
    with open(save_root / "test_metrics_summary.json", "w") as f:
        json.dump({"means": means, "stds": stds, "metric_order": metric_names}, f, indent=2)

    # Single aggregated plot (avg + std cloud) for train/val LOSS
    plot_path = save_root / "avg_train_val_loss.png"
    plot_avg_std(train_curves, val_curves, plot_path)
    print(f"\nSaved aggregated train/val loss plot -> {plot_path.resolve()}")

if __name__ == "__main__":
    main()
