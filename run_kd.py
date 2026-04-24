"""
run_kd.py — Multi-Teacher KD 학습 스크립트
==============================================

하나의 (α, seed, weighting) 조합에 대해:
  1. Teacher logit 및 Expertise Matrix 로드
  2. Weighting 방법으로 가중치 산출
  3. Student (ResNet-18 scratch)를 KD로 학습
  4. 평가 결과 저장

실행:
  python run_kd.py --alpha 0.1 --seed 42 --weighting uniform
  python run_kd.py --alpha 0.1 --seed 42 --weighting top_1
  python run_kd.py --alpha 0.1 --seed 42 --weighting top_3

EC2 분산 예시 (30개 조합):
  EC2 A: α=0.1, seed=42, weighting={uniform, top_1, top_3}
  EC2 B: α=0.5, seed=42, weighting={uniform, top_1, top_3}
  ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from common import (
    ensure_dirs,
    partition_path, teachers_dir, kd_dir, bounds_dir, logs_dir,
    set_seed, EpochTimer,
    ParquetImageDataset, load_parquet_table, get_transforms, IndexedSubset,
    build_resnet18, evaluate,
    save_json,
)


# =============================================================================
# 하이퍼파라미터
# =============================================================================
EPOCHS_KD = 80
BATCH_SIZE = 128
LR = 0.1
WD = 5e-4
MOMENTUM = 0.9
NUM_WORKERS = 8
NUM_CLASSES = 100
NUM_CLIENTS = 5

KD_ALPHA = 0.5        # CE vs KD 비율 (loss = α*CE + (1-α)*KD)
KD_TEMPERATURE = 4.0  # softmax temperature


# =============================================================================
# 가중치 산출
# =============================================================================
def compute_class_weights(expertise_f1: np.ndarray, method: str,
                          k_select: int = 3):
    """클래스별 Teacher 가중치 산출.

    Args:
      expertise_f1: (K, C) — per-class F1
      method:  'uniform' | 'top_1' | 'top_3'

    Returns:
      weights: (K, C) — 열합=1 (클래스별로 정규화)
    """
    K, C = expertise_f1.shape
    weights = np.zeros((K, C))

    if method == "uniform":
        weights[:, :] = 1.0 / K

    elif method in ("top_1", "top_3"):
        k_sel = 1 if method == "top_1" else 3
        for c in range(C):
            top_idx = np.argsort(expertise_f1[:, c])[-k_sel:]
            scores = expertise_f1[top_idx, c]
            total = scores.sum()
            if total > 0:
                weights[top_idx, c] = scores / total
            else:
                # 이 클래스를 아는 전문가가 아무도 없음 → 균등
                weights[:, c] = 1.0 / K
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return weights


def fuse_logits(teacher_logits_dict, weights):
    """Teacher logit들을 클래스별 가중치로 합산.

    Args:
      teacher_logits_dict: {k: (N, C)}
      weights:            (K, C)

    Returns:
      fused: (N, C) torch.Tensor
    """
    K = len(teacher_logits_dict)
    N, C = teacher_logits_dict[0].shape
    fused = torch.zeros(N, C)
    weights_t = torch.from_numpy(weights).float()  # (K, C)

    for k in range(K):
        # weights_t[k]는 (C,) — broadcasting으로 (N,C) * (C,) 적용
        fused += teacher_logits_dict[k] * weights_t[k].unsqueeze(0)
    return fused


# =============================================================================
# KD Loss
# =============================================================================
def kd_loss(student_logits, teacher_logits, T: float):
    """KL-divergence KD loss (Hinton 2015, T² 보정 포함)."""
    s_log = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s_log, t, reduction="batchmean") * (T * T)


# =============================================================================
# KD 학습 루프
# =============================================================================
def train_student_kd(fused_logits, proxy_loader_aug, val_loader,
                     epochs, device, seed, tag, use_amp=True,
                     log_path=None):
    """Student를 scratch에서 fused teacher logit으로 KD 학습."""
    set_seed(seed * 7919)  # student용 별도 파생 seed

    student = build_resnet18(num_classes=NUM_CLASSES, pretrained=False).to(device)
    optimizer = torch.optim.SGD(
        student.parameters(), lr=LR, momentum=MOMENTUM,
        weight_decay=WD, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    timer = EpochTimer(tag=tag, log_path=log_path)

    fused_on_dev = fused_logits.to(device)  # (N_proxy, C) — 인덱스로 조회

    best_acc = 0.0
    best_state = None
    best_epoch = -1
    history = []

    for epoch in range(epochs):
        with timer.epoch(epoch):
            student.train()
            total_loss, total_ce, total_kd = 0.0, 0.0, 0.0
            correct, total = 0, 0

            for imgs, labels, pos in proxy_loader_aug:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # fused logit 조회 — pos가 proxy 내 순서
                t_logits = fused_on_dev[pos.to(device)]

                optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        s_logits = student(imgs)
                        loss_ce = ce_criterion(s_logits, labels)
                        loss_kd = kd_loss(s_logits, t_logits, KD_TEMPERATURE)
                        loss = KD_ALPHA * loss_ce + (1 - KD_ALPHA) * loss_kd
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    s_logits = student(imgs)
                    loss_ce = ce_criterion(s_logits, labels)
                    loss_kd = kd_loss(s_logits, t_logits, KD_TEMPERATURE)
                    loss = KD_ALPHA * loss_ce + (1 - KD_ALPHA) * loss_kd
                    loss.backward()
                    optimizer.step()

                bs = imgs.size(0)
                total_loss += loss.item() * bs
                total_ce += loss_ce.item() * bs
                total_kd += loss_kd.item() * bs
                correct += (s_logits.argmax(1) == labels).sum().item()
                total += bs

            scheduler.step()

        avg_loss = total_loss / max(total, 1)
        avg_ce = total_ce / max(total, 1)
        avg_kd = total_kd / max(total, 1)
        tr_acc = correct / max(total, 1)

        # 검증은 매 2에폭 + 마지막 10에폭
        do_eval = (epoch % 2 == 0) or (epoch >= epochs - 10)
        if do_eval:
            val_m = evaluate(student, val_loader, device, NUM_CLASSES)
            val_acc = val_m["accuracy"]
            record = {
                "epoch": epoch, "loss": avg_loss,
                "ce": avg_ce, "kd": avg_kd, "train_acc": tr_acc,
                "val_acc": val_acc, "macro_f1": val_m["macro"]["f1"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(f"  [{tag}] epoch {epoch:3d} | loss={avg_loss:.3f} "
                  f"(ce={avg_ce:.3f}, kd={avg_kd:.3f}) "
                  f"| tr_acc={tr_acc:.4f} | val_acc={val_acc:.4f}",
                  flush=True)

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

    timer.summary()
    return best_state, best_acc, best_epoch, history, timer.summary()


# =============================================================================
# 메인
# =============================================================================
def run_kd(alpha: float, seed: int, weighting: str,
           use_amp: bool = True, force: bool = False):
    out_dir = kd_dir(alpha, seed, weighting)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "student_best.pt"
    metrics_path = out_dir / "metrics.json"

    if metrics_path.exists() and not force:
        print(f"[α={alpha}, s={seed}, w={weighting}] SKIP — 이미 존재")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================
    # 파티션 / Teacher / Expertise 로드
    # =====================================
    p_path = partition_path(alpha, seed)
    t_dir = teachers_dir(alpha, seed)

    if not p_path.exists():
        raise FileNotFoundError(f"파티션 없음: {p_path}")
    if not (t_dir / "teacher_logits.pt").exists():
        raise FileNotFoundError(
            f"Teacher logit 없음: {t_dir}/teacher_logits.pt. "
            f"먼저 train_teachers.py를 실행하세요."
        )

    data = np.load(p_path)
    proxy_indices = data["proxy_indices"]

    tl_bundle = torch.load(t_dir / "teacher_logits.pt",
                           map_location="cpu", weights_only=False)
    teacher_logits = tl_bundle["teacher_logits"]
    proxy_labels = tl_bundle["proxy_labels"]

    expertise = np.load(t_dir / "expertise.npz")
    f1_mat = expertise["f1"]  # (K, C)

    # =====================================
    # 가중치 산출 + logit fusion
    # =====================================
    print(f"\n===== α={alpha}, seed={seed}, weighting={weighting} =====")
    weights = compute_class_weights(f1_mat, weighting)
    fused = fuse_logits(teacher_logits, weights)
    print(f"  Fused logit shape: {tuple(fused.shape)}")

    # 가중치 분포 요약
    nonzero_per_class = (weights > 0).sum(axis=0)  # 각 클래스에 기여하는 Teacher 수
    print(f"  클래스당 기여 Teacher 수 분포: "
          f"min={int(nonzero_per_class.min())}, "
          f"max={int(nonzero_per_class.max())}, "
          f"mean={float(nonzero_per_class.mean()):.2f}")

    # =====================================
    # Proxy DataLoader (augmented, shuffle=True, IndexedSubset)
    # Parquet 기반: train은 한 번만 로드
    # =====================================
    train_ds_aug = ParquetImageDataset(
        transform=get_transforms(True), shared_table=load_parquet_table("train"))
    val_ds = ParquetImageDataset(
        transform=get_transforms(False), shared_table=load_parquet_table("val"))

    indexed = IndexedSubset(train_ds_aug, proxy_indices)
    proxy_loader_aug = DataLoader(
        indexed, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # =====================================
    # KD 학습
    # =====================================
    tag = f"kd_a{alpha}_s{seed}_{weighting}"
    log_path = logs_dir() / f"{tag}.log"

    best_state, best_acc, best_epoch, history, tsum = train_student_kd(
        fused, proxy_loader_aug, val_loader, EPOCHS_KD, device,
        seed, tag, use_amp=use_amp, log_path=log_path)

    # best state 저장
    torch.save({
        "student_state": best_state,
        "best_epoch": best_epoch,
        "best_acc": best_acc,
        "alpha": alpha, "seed": seed, "weighting": weighting,
    }, ckpt_path)

    # =====================================
    # 최종 평가 + Gap Recovery
    # =====================================
    student = build_resnet18(num_classes=NUM_CLASSES, pretrained=False).to(device)
    student.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final = evaluate(student, val_loader, device, NUM_CLASSES)

    # Lower/Upper Bound 로드 (있으면)
    gap_recovery = None
    lower_acc, upper_acc = None, None
    lower_p = bounds_dir(seed) / "lower_metrics.json"
    upper_p = bounds_dir(seed) / "upper_metrics.json"
    if lower_p.exists() and upper_p.exists():
        with open(lower_p) as f:
            lower_acc = json.load(f)["final_accuracy"]
        with open(upper_p) as f:
            upper_acc = json.load(f)["final_accuracy"]
        gap = upper_acc - lower_acc
        if gap > 0:
            gap_recovery = (final["accuracy"] - lower_acc) / gap * 100
            print(f"\n  Lower={lower_acc:.4f}, Upper={upper_acc:.4f}, "
                  f"Gap={gap:.4f}")
            print(f"  Student={final['accuracy']:.4f} → "
                  f"Gap Recovery = {gap_recovery:.1f}%")

    # =====================================
    # 결과 저장
    # =====================================
    result = {
        "alpha": alpha, "seed": seed, "weighting": weighting,
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "final_accuracy": final["accuracy"],
        "final_macro_f1": final["macro"]["f1"],
        "per_class": final["per_class"],
        "history": history,
        "lower_acc": lower_acc,
        "upper_acc": upper_acc,
        "gap_recovery_pct": gap_recovery,
        "coverage_per_class": nonzero_per_class.tolist(),
        "kd_alpha": KD_ALPHA,
        "kd_temperature": KD_TEMPERATURE,
        "epochs": EPOCHS_KD,
        "time_summary": tsum,
    }
    save_json(result, metrics_path)

    print(f"\n===== DONE =====")
    print(f"  Final acc    : {final['accuracy']:.4f}")
    print(f"  Best epoch   : {best_epoch}")
    if gap_recovery is not None:
        print(f"  Gap Recovery : {gap_recovery:.1f}%")
    print(f"  saved        → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--weighting", choices=["uniform", "top_1", "top_3"],
                        required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    run_kd(args.alpha, args.seed, args.weighting,
           use_amp=not args.no_amp, force=args.force)


if __name__ == "__main__":
    main()
