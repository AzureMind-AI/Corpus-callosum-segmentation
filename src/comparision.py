from operator import le
import os
import math
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt

# ---- local modules ----
from datasets import Dataset_MRI_2D_Fold
from models_baseline import UNet_baseline, UNetPlusPlus
from models import UNet_Enhanced

# =========================
# Config (base config, will be overridden per model)
# =========================
BASE_CONFIG = {
    "LEARNING_RATE": 2e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "BATCH_SIZE": 4,
    "NUM_EPOCHS": 30,
    "IMAGE_HEIGHT": 512,
    "IMAGE_WIDTH": 512,
    "IN_CHANNELS": 1,
    "NUM_CLASSES": 1,

    "TRAIN_IMG_DIR": "data/train/images",
    "TRAIN_LBL_DIR": "data/train/labels",
    "TEST_IMG_DIR": "data/test/images",
    "TEST_LBL_DIR": "data/test/labels",

    "K_FOLDS": 10,
    "NUM_WORKERS": 4,
    "USE_MULTI_GPU": True,
    "SAVE_DIR": "runs_baseline",  # Base dir, will append model name
    "SEED": 42,

    # Deep supervision weights (used if applicable, but for baselines we use average)
    "DEEP_SUPERVISION_WEIGHTS": [1.0, 0.8, 0.6, 0.4],
}

# Models to run
MODELS_TO_RUN = [
    # {"name": "UNetPlusPlus", "use_ds": True},
    # {"name": "UNet_baseline", "use_ds": False},
    {"name": "AURANET", "use_ds": True},
]

# =========================
# Utilities (same as main.py)
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
    if n > 1 and BASE_CONFIG["USE_MULTI_GPU"]:
        print(f"Using {n} GPUs (DataParallel).")
        return "cuda", n, n
    print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    return "cuda", 1, 1

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

def train_one_epoch(model, loader, optimizer, loss_fn, device, use_deep_supervision=False):
    model.train()
    loop = tqdm(loader, leave=False, desc="train")
    total = 0.0

    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        if use_deep_supervision:
            out = model(images)
        else:
            out = model(images)

        if isinstance(out, tuple):
            # Average losses for baselines (unlike weighted in original)
            losses = [loss_fn(p, masks) for p in out]
            loss = sum(losses) / len(losses)
        else:
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
        
        # Handle tuple output from deep supervision
        if isinstance(out, tuple):
            # Average losses for deep supervision outputs
            losses = [loss_fn(p, masks) for p in out]
            loss = sum(losses) / len(losses)
            # Use final output (last element) for metrics
            out_for_metrics = out[-1]
        else:
            loss = loss_fn(out, masks)
            out_for_metrics = out
        
        tot_loss += loss.item()

        m = all_metrics(out_for_metrics, masks)
        for k,v in m.items():
            if not (np.isnan(v) or np.isinf(v)):
                agg[k].append(v)

        loop.set_postfix(loss=loss.item(), dice=(agg["dice"][-1] if agg["dice"] else 0.0))

    avg = {k: (float(np.mean(v)) if len(v)>0 else 0.0) for k,v in agg.items()}
    return tot_loss / max(1, len(loader)), avg

def plot_avg_std(train_curves: List[List[float]], val_curves: List[List[float]], save_path: Path):
    arr_train = np.array(train_curves, dtype=float)
    arr_val   = np.array(val_curves, dtype=float)

    mean_train = arr_train.mean(axis=0)
    std_train  = arr_train.std(axis=0)
    mean_val   = arr_val.mean(axis=0)
    std_val    = arr_val.std(axis=0)

    epochs = np.arange(1, mean_train.shape[0]+1)

    plt.figure(figsize=(9,6))
    plt.plot(epochs, mean_train, label="Train loss")
    plt.plot(epochs, mean_val,   label="Val loss")

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

def run_one_fold(
    fold_idx: int,
    master: Dataset_MRI_2D_Fold,
    image_transform,
    mask_transform,
    device: str,
    gpu_mult: int,
    save_root: Path,
    config: dict,
) -> Tuple[Dict[str, float], List[float], List[float]]:
    """
    Performs 10-fold cross-validation on the training portion of the fold,
    averages curves, and evaluates the best-performing model on the test set.
    """

    # Get train/test for current outer fold
    fold_train_ds, fold_test_ds = master.get_fold_datasets(fold_idx)


    kf = KFold(n_splits=10, shuffle=True, random_state=config["SEED"] + fold_idx)
    all_indices = np.arange(len(fold_train_ds))

    eff_batch = config["BATCH_SIZE"] * gpu_mult
    eff_workers = config["NUM_WORKERS"] * gpu_mult if device == "cuda" and gpu_mult > 1 else config["NUM_WORKERS"]

    fold_dir = save_root / f"fold_{fold_idx+1:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    best_val_dice = -1.0
    best_state = None
    best_inner_fold = -1

    all_train_curves = []
    all_val_curves = []

    # ---------- Inner 10-Fold CV ----------
    for inner_idx, (train_idx, val_idx) in enumerate(kf.split(all_indices)):

        inner_train_ds = torch.utils.data.Subset(fold_train_ds, train_idx)
        inner_val_ds = torch.utils.data.Subset(fold_train_ds, val_idx)
        train_loader = DataLoader(inner_train_ds, batch_size=eff_batch, shuffle=True,
                                  num_workers=eff_workers, pin_memory=(device=="cuda"))
        val_loader   = DataLoader(inner_val_ds,   batch_size=eff_batch, shuffle=False,
                                  num_workers=eff_workers, pin_memory=(device=="cuda"))
        if len(train_loader)<2:
            continue
        # Model
        model_name = config["MODEL_NAME"]
        if model_name == "UNet_baseline":
            model = UNet_baseline(in_channels=config["IN_CHANNELS"], n_classes=config["NUM_CLASSES"])
        elif model_name == "UNetPlusPlus":
            model = UNetPlusPlus(
                in_channels=config["IN_CHANNELS"],
                n_classes=config["NUM_CLASSES"],
                deep_supervision=config["USE_DEEP_SUPERVISION"]
            )
        elif model_name == "AURANET":
            model = UNet_Enhanced(in_channels=config["IN_CHANNELS"], n_classes=config["NUM_CLASSES"], deep_supervision=config["USE_DEEP_SUPERVISION"])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        if device == "cuda":
            model = model.cuda()
            if torch.cuda.device_count() > 1 and config["USE_MULTI_GPU"]:
                model = torch.nn.DataParallel(model)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"] * gpu_mult)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=False)

        train_curve, val_curve = [], []
        best_val_dice_inner = -1.0
        best_state_inner = None

        for epoch in range(config["NUM_EPOCHS"]):
            model.train()
            tr_loss = train_one_epoch(
                model, train_loader, optimizer, loss_fn, device,
                use_deep_supervision=config["USE_DEEP_SUPERVISION"],
            )
            model.eval()
            va_loss, va_metrics = evaluate(model, val_loader, loss_fn, device)
            scheduler.step(va_loss)

            cur_val_dice = va_metrics.get("dice", 0.0)
            train_curve.append(tr_loss)
            val_curve.append(va_loss)


            if cur_val_dice > best_val_dice_inner:
                best_val_dice_inner = cur_val_dice
                best_state_inner = (model.module.state_dict() if isinstance(model, torch.nn.DataParallel)
                                    else model.state_dict())

        all_train_curves.append(train_curve)
        all_val_curves.append(val_curve)

        if best_val_dice_inner > best_val_dice:
            best_val_dice = best_val_dice_inner
            best_state = best_state_inner
            best_inner_fold = inner_idx + 1

        del model
   
    # ---------- Average Curves ----------
    max_epochs = config["NUM_EPOCHS"]
    def pad_and_average(curves):
        padded = np.array([np.pad(c, (0, max_epochs - len(c)), mode='edge') for c in curves])
        return list(padded.mean(axis=0))
    avg_train_curve = pad_and_average(all_train_curves)
    avg_val_curve = pad_and_average(all_val_curves)

    print(f"\n[Outer Fold {fold_idx+1:02d}] Best Inner Fold = {best_inner_fold}, Best Val Dice = {best_val_dice:.4f}")

    # ---------- Evaluate on TEST ----------
    if best_state is None:
        raise RuntimeError("No best model found during inner folds!")

    # Rebuild model for testing
    if model_name == "UNet_baseline":
        model = UNet_baseline(in_channels=config["IN_CHANNELS"], n_classes=config["NUM_CLASSES"])
    elif model_name == "UNetPlusPlus":
        model = UNetPlusPlus(
            in_channels=config["IN_CHANNELS"],
            n_classes=config["NUM_CLASSES"],
            deep_supervision=False
        )
    elif model_name == "AURANET":
        model = UNet_Enhanced(in_channels=config["IN_CHANNELS"], n_classes=config["NUM_CLASSES"], deep_supervision=False )
    if device == "cuda":
        model = model.cuda()
        if torch.cuda.device_count() > 1 and config["USE_MULTI_GPU"]:
            model = torch.nn.DataParallel(model)

    model.load_state_dict(best_state)
    model.eval()

    test_loader = DataLoader(fold_test_ds, batch_size=eff_batch, shuffle=False,
                             num_workers=eff_workers, pin_memory=(device=="cuda"))
    test_loss, test_metrics = evaluate(model, test_loader, loss_fn, device)

    print(f"[FOLD {fold_idx+1:02d}] TEST -> loss={test_loss:.4f}, "
          f"dice={test_metrics['dice']:.4f}, jaccard={test_metrics['jaccard']:.4f}, "
          f"acc={test_metrics['accuracy']:.4f}, sens={test_metrics['sensitivity']:.4f}, "
          f"spec={test_metrics['specificity']:.4f}, hd={test_metrics['hausdorff']:.4f}")

    with open(fold_dir / "test_metrics.json", "w") as f:
        json.dump({
            "test_loss": test_loss,
            **test_metrics,
            "best_val_dice": best_val_dice,
            "best_inner_fold": best_inner_fold,
            "avg_train_curve": avg_train_curve,
            "avg_val_curve": avg_val_curve,
        }, f, indent=2)

    return test_metrics, avg_train_curve, avg_val_curve


# =========================
# Main logic for each model
# =========================
def run_for_model(model_cfg: dict, device, n_gpus, gpu_mult):
    config = BASE_CONFIG.copy()
    config["MODEL_NAME"] = model_cfg["name"]
    config["USE_DEEP_SUPERVISION"] = model_cfg["use_ds"]
    save_root = Path(config["SAVE_DIR"]) / config["MODEL_NAME"]
    save_root.mkdir(parents=True, exist_ok=True)

    per_fold_metrics: List[Dict[str,float]] = []
    train_curves: List[List[float]] = []
    val_curves: List[List[float]] = []
    if config["MODEL_NAME"] == "AURANET":
        image_transform = transforms.Compose([
            transforms.Resize((BASE_CONFIG["IMAGE_HEIGHT"], BASE_CONFIG["IMAGE_WIDTH"]), antialias=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        image_transform = transforms.Compose([
            transforms.Resize((BASE_CONFIG["IMAGE_HEIGHT"], BASE_CONFIG["IMAGE_WIDTH"]), antialias=True)
        ])

    mask_transform = transforms.Compose([
        transforms.Resize((BASE_CONFIG["IMAGE_HEIGHT"], BASE_CONFIG["IMAGE_WIDTH"]), antialias=True),
    ])
    master = Dataset_MRI_2D_Fold(
        train_image_dir=BASE_CONFIG["TRAIN_IMG_DIR"],
        train_label_dir=BASE_CONFIG["TRAIN_LBL_DIR"],
        test_image_dir=BASE_CONFIG["TEST_IMG_DIR"],
        test_label_dir=BASE_CONFIG["TEST_LBL_DIR"],
        image_transform=image_transform,
        mask_transform=mask_transform,
        k_folds=BASE_CONFIG["K_FOLDS"],
        verbose=True
    )

    for f in range(config["K_FOLDS"]):
        m, tr_c, va_c = run_one_fold(
            fold_idx=f,
            master=master,
            image_transform=image_transform,
            mask_transform=mask_transform,
            device=device,
            gpu_mult=gpu_mult,
            save_root=save_root,
            config=config
        )
        per_fold_metrics.append(m)
        train_curves.append(tr_c)
        val_curves.append(va_c)

    # Summaries (same as main.py)
    metric_names = ["dice","jaccard","accuracy","sensitivity","specificity","hausdorff"]
    means = {}
    stds  = {}

    print(f"\n========== PER-FOLD TEST METRICS for {config['MODEL_NAME']} ==========")
    for i, m in enumerate(per_fold_metrics, start=1):
        print(f"Fold {i:02d}: " +
              ", ".join([f"{k}={m.get(k, 0.0):.4f}" for k in metric_names]))

    print(f"\n============= SUMMARY (TEST) for {config['MODEL_NAME']} =============")
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

        df_summary = pd.DataFrame([{"metric": k, "mean": means[k], "std": stds[k]} for k in metric_names])
        df_summary.to_csv(save_root / "test_metrics_summary.csv", index=False)
    except Exception as e:
        print("Pandas not available? Skipping CSV save.", e)

    # Save JSON summary
    with open(save_root / "test_metrics_summary.json", "w") as f:
        json.dump({"means": means, "stds": stds, "metric_order": metric_names}, f, indent=2)

    # Aggregated plot
    plot_path = save_root / "avg_train_val_loss.png"
    plot_avg_std(train_curves, val_curves, plot_path)
    print(f"\nSaved aggregated train/val loss plot for {config['MODEL_NAME']} -> {plot_path.resolve()}")

# =========================
# Main
# =========================
def main():
    set_seed(BASE_CONFIG["SEED"])
    device, n_gpus, gpu_mult = setup_multi_gpu()



    for model_cfg in MODELS_TO_RUN:
        print(f"\n\n=== Running experiments for {model_cfg['name']} (DS: {model_cfg['use_ds']}) ===")
        run_for_model(model_cfg, device, n_gpus, gpu_mult)

if __name__ == "__main__":
    main()