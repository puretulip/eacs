"""
train_bounds.py — Lower / Upper Bound 학습 스크립트
======================================================

Lower Bound: proxy data만으로 Student(ResNet-18) CE 학습
Upper Bound: 전체 train data로 ResNet-18 CE 학습 (centralized)

둘 다 α와 무관하므로 seed당 1번씩만 돌리면 됨.

실행:
  python train_bounds.py --seed 42 --mode lower
  python train_bounds.py --seed 42 --mode upper
  python train_bounds.py --seed 42 --mode both

EC2 분산 시:
  EC2 A: python train_bounds.py --seed 42 --mode both
  EC2 B: python train_bounds.py --seed 123 --mode both
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from common import (
    DATA_ROOT, ensure_dirs,
    partition_path, bounds_dir, logs_dir,
    set_seed, EpochTimer,
    ImageNet100Dataset, get_transforms,
    build_resnet18, train_one_epoch, evaluate,
    save_json,
)


# =============================================================================
# 하이퍼파라미터 (Bounds용)
# =============================================================================
EPOCHS_LOWER = 100      # proxy만이라 더 많은 에폭 필요
EPOCHS_UPPER = 60       # 전체 데이터, 더 적은 에폭으로도 수렴
BATCH_SIZE = 128
LR = 0.1
WD = 5e-4
MOMENTUM = 0.9
NUM_WORKERS = 8
NUM_CLASSES = 100


def train_bound(mode: str, seed: int, use_amp: bool = True):
    """mode in {'lower', 'upper'}."""
    assert mode in ("lower", "upper")

    out_dir = bounds_dir(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{mode}.pt"
    metrics_path = out_dir / f"{mode}_metrics.json"

    if ckpt_path.exists():
        print(f"[{mode} seed={seed}] SKIP — 이미 존재: {ckpt_path}")
        return

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================
    # 데이터 준비
    # =====================================
    train_ds_full = ImageNet100Dataset("train", transform=get_transforms(True))
    val_ds = ImageNet100Dataset("val", transform=get_transforms(False))

    if mode == "lower":
        # proxy만 사용 — 파티션은 seed만 따라감 (α 무관)
        # 대표로 alpha=1.0 파티션의 proxy_indices를 사용해도 무방 (proxy는 α와 무관)
        # 단, 파티션이 없으면 prepare_partition.py를 먼저 돌려야 함
        p_path = partition_path(1.0, seed)
        if not p_path.exists():
            raise FileNotFoundError(
                f"파티션이 없습니다: {p_path}. "
                f"먼저 `python prepare_partition.py --alpha 1.0 --seed {seed}` 실행하세요."
            )
        data = np.load(p_path)
        proxy_idx = data["proxy_indices"]
        train_ds = Subset(train_ds_full, proxy_idx.tolist())
        epochs = EPOCHS_LOWER
        print(f"[lower] proxy samples: {len(train_ds):,}")
    else:
        train_ds = train_ds_full
        epochs = EPOCHS_UPPER
        print(f"[upper] full train samples: {len(train_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    # =====================================
    # 모델 / 옵티마이저
    # =====================================
    model = build_resnet18(num_classes=NUM_CLASSES, pretrained=False).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    timer = EpochTimer(
        tag=f"{mode}_seed{seed}",
        log_path=logs_dir() / f"bounds_{mode}_seed{seed}.log"
    )

    # =====================================
    # 학습 루프
    # =====================================
    best_acc = 0.0
    best_epoch = -1
    history = []

    for epoch in range(epochs):
        with timer.epoch(epoch):
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler)
            scheduler.step()

        # 검증은 매 5에폭 + 마지막 10에폭
        do_eval = (epoch % 5 == 0) or (epoch >= epochs - 10)
        val_metrics = None
        if do_eval:
            val_metrics = evaluate(model, val_loader, device, NUM_CLASSES)
            val_acc = val_metrics["accuracy"]
            record = {
                "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                "val_acc": val_acc, "val_macro_f1": val_metrics["macro"]["f1"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(f"  [{mode}] epoch {epoch:3d} | train_acc={tr_acc:.4f} "
                  f"| val_acc={val_acc:.4f} | macro_f1={val_metrics['macro']['f1']:.4f}",
                  flush=True)

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "mode": mode,
                    "seed": seed,
                }, ckpt_path)

    timer.summary()

    # =====================================
    # 최종 평가 (best model 로드)
    # =====================================
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    final = evaluate(model, val_loader, device, NUM_CLASSES)

    result = {
        "mode": mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "final_accuracy": final["accuracy"],
        "final_macro_f1": final["macro"]["f1"],
        "per_class": final["per_class"],
        "history": history,
        "n_train_samples": len(train_ds),
        "epochs_trained": epochs,
        "time_summary": timer.summary(),
    }
    save_json(result, metrics_path)
    print(f"\n[{mode} seed={seed}] DONE | best_acc={best_acc:.4f} @ epoch {best_epoch}")
    print(f"  ckpt    → {ckpt_path}")
    print(f"  metrics → {metrics_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["lower", "upper", "both"], default="both")
    parser.add_argument("--no-amp", action="store_true",
                        help="Mixed precision 비활성화")
    args = parser.parse_args()

    ensure_dirs()

    modes = ["lower", "upper"] if args.mode == "both" else [args.mode]
    for m in modes:
        train_bound(m, args.seed, use_amp=not args.no_amp)


if __name__ == "__main__":
    main()
