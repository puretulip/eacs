"""
train_teachers.py — Teacher 학습 + Logit/Quality 사전 수집
=============================================================

하나의 (α, seed) 조합에 대해:
  1. K=5명의 Teacher를 독립 학습 (각자의 private data로)
  2. Proxy data에 대한 logit 사전 수집 (KD 재학습 시 재사용)
  3. Expertise Matrix (per-class F1/Precision/Recall) 계산
  4. Layer 2 지표 (logit entropy, top-1 conf 등) 계산
  5. 결과를 teachers_dir로 저장

실행:
  python train_teachers.py --alpha 0.1 --seed 42
  python train_teachers.py --alpha 0.5 --seed 42
  ...

EC2 분산 예시 (10개 조합):
  EC2 A: α={0.1, 0.5},   seed=42
  EC2 B: α={1.0, 10.0},  seed=42
  EC2 C: α={100.0},      seed=42, 그리고 α={0.1}, seed=123
  ... 원하는 방식으로 쪼개기
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from common import (
    ensure_dirs,
    partition_path, teachers_dir, logs_dir,
    set_seed, EpochTimer,
    ImageNet100Dataset, get_transforms,
    build_resnet18, train_one_epoch, evaluate,
    collect_logits, logit_quality_metrics,
    save_json,
)


# =============================================================================
# 하이퍼파라미터
# =============================================================================
EPOCHS_TEACHER = 60
BATCH_SIZE = 128
LR = 0.1
WD = 5e-4
MOMENTUM = 0.9
NUM_WORKERS = 8
NUM_CLASSES = 100
NUM_CLIENTS = 5


def train_single_teacher(k: int, indices: np.ndarray, train_ds_full,
                         val_loader, device, seed: int, alpha: float,
                         use_amp: bool = True):
    """단일 Teacher 학습."""
    # seed를 (k, seed)로 파생시켜 Teacher 간 다른 초기화
    set_seed(seed * 1000 + k)

    # 데이터
    if len(indices) < BATCH_SIZE:
        print(f"  [WARN] Teacher {k}: only {len(indices)} samples, "
              f"batch size 감소")
        bs = max(16, len(indices) // 4)
    else:
        bs = BATCH_SIZE

    subset = Subset(train_ds_full, indices.tolist())
    loader = DataLoader(subset, batch_size=bs, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        drop_last=(len(subset) >= bs * 2))

    # 모델
    model = build_resnet18(num_classes=NUM_CLASSES, pretrained=False).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_TEACHER)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    timer = EpochTimer(
        tag=f"teacher_k{k}_a{alpha}_s{seed}",
        log_path=logs_dir() / f"teacher_a{alpha}_s{seed}_k{k}.log"
    )

    for epoch in range(EPOCHS_TEACHER):
        with timer.epoch(epoch):
            tr_loss, tr_acc = train_one_epoch(
                model, loader, optimizer, criterion, device, scaler)
            scheduler.step()

        if epoch % 10 == 0 or epoch == EPOCHS_TEACHER - 1:
            val_metrics = evaluate(model, val_loader, device, NUM_CLASSES)
            print(f"  [T{k}] epoch {epoch:3d} | tr_acc={tr_acc:.4f} "
                  f"| val_acc={val_metrics['accuracy']:.4f} "
                  f"| macro_f1={val_metrics['macro']['f1']:.4f}", flush=True)

    timer.summary()
    final_val = evaluate(model, val_loader, device, NUM_CLASSES)
    return model, final_val, timer.summary()


def run_teachers(alpha: float, seed: int, use_amp: bool = True,
                 force: bool = False):
    """하나의 (α, seed) 조합에 대해 K명의 Teacher 학습 + logit 수집."""
    out_dir = teachers_dir(alpha, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    teachers_path = out_dir / "teachers.pt"
    logits_path = out_dir / "teacher_logits.pt"
    expertise_path = out_dir / "expertise.npz"
    meta_path = out_dir / "metadata.json"

    if teachers_path.exists() and logits_path.exists() and not force:
        print(f"[α={alpha}, seed={seed}] SKIP — 이미 존재")
        return

    # =====================================
    # 파티션 로드
    # =====================================
    p_path = partition_path(alpha, seed)
    if not p_path.exists():
        raise FileNotFoundError(
            f"파티션이 없습니다: {p_path}. "
            f"먼저 `python prepare_partition.py --alpha {alpha} --seed {seed}` 실행.")

    data = np.load(p_path)
    proxy_indices = data["proxy_indices"]
    client_indices = {k: data[f"client_{k}"] for k in range(NUM_CLIENTS)}

    print(f"\n===== α={alpha}, seed={seed} =====")
    print(f"  Proxy:   {len(proxy_indices):,}")
    for k in range(NUM_CLIENTS):
        print(f"  Client {k}: {len(client_indices[k]):,}장")

    # =====================================
    # 데이터셋 / DataLoaders
    # =====================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds_full = ImageNet100Dataset("train", transform=get_transforms(True))
    val_ds = ImageNet100Dataset("val", transform=get_transforms(False))
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # proxy (no-aug) loader — logit 수집용
    train_ds_noaug = ImageNet100Dataset("train", transform=get_transforms(False))
    proxy_subset = Subset(train_ds_noaug, proxy_indices.tolist())
    proxy_loader_noaug = DataLoader(
        proxy_subset, batch_size=256, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    # =====================================
    # Teacher 학습
    # =====================================
    teacher_states = {}
    teacher_val_metrics = {}
    time_summaries = {}

    for k in range(NUM_CLIENTS):
        print(f"\n--- Teacher {k} 학습 ---")
        model, val_m, tsum = train_single_teacher(
            k, client_indices[k], train_ds_full, val_loader, device,
            seed, alpha, use_amp=use_amp)
        teacher_states[k] = {kk: v.cpu() for kk, v in model.state_dict().items()}
        teacher_val_metrics[k] = {
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro"]["f1"],
        }
        time_summaries[f"teacher_{k}"] = tsum
        # Teacher state는 한 명 끝날 때마다 임시 저장 (crash 대비)
        torch.save(teacher_states, teachers_path)

    # =====================================
    # Proxy logit 사전 수집
    # =====================================
    print(f"\n--- Proxy logit 수집 (Teachers → proxy) ---")
    teacher_logits = {}
    proxy_labels = None

    for k in range(NUM_CLIENTS):
        model = build_resnet18(num_classes=NUM_CLASSES, pretrained=False).to(device)
        model.load_state_dict({kk: v.to(device) for kk, v in teacher_states[k].items()})
        logits, labels = collect_logits(model, proxy_loader_noaug, device)
        teacher_logits[k] = logits
        if proxy_labels is None:
            proxy_labels = labels
        print(f"  T{k}: logits shape={tuple(logits.shape)}, "
              f"top-1 on proxy={float((logits.argmax(1)==labels).float().mean()):.4f}")

    torch.save({
        "teacher_logits": teacher_logits,
        "proxy_labels": proxy_labels,
    }, logits_path)

    # =====================================
    # Expertise Matrix (per-class F1/P/R)
    # =====================================
    print(f"\n--- Expertise Matrix 계산 ---")
    f1_mat = np.zeros((NUM_CLIENTS, NUM_CLASSES))
    precision_mat = np.zeros((NUM_CLIENTS, NUM_CLASSES))
    recall_mat = np.zeros((NUM_CLIENTS, NUM_CLASSES))
    labels_np = proxy_labels.numpy()

    for k in range(NUM_CLIENTS):
        preds = teacher_logits[k].argmax(dim=1).numpy()
        for c in range(NUM_CLASSES):
            true_c = (labels_np == c)
            pred_c = (preds == c)
            tp = int((true_c & pred_c).sum())
            fp = int((~true_c & pred_c).sum())
            fn = int((true_c & ~pred_c).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            precision_mat[k, c] = prec
            recall_mat[k, c] = rec
            f1_mat[k, c] = f1

    np.savez(expertise_path,
             f1=f1_mat, precision=precision_mat, recall=recall_mat)

    # Expertise 요약 출력
    print(f"  {'Teacher':<10} {'F1>0.5':>8} {'best F1':>10} "
          f"{'mean F1':>10} {'proxy acc':>10}")
    print(f"  {'-'*55}")
    for k in range(NUM_CLIENTS):
        f1_s = f1_mat[k]
        proxy_acc = float((teacher_logits[k].argmax(1) == proxy_labels).float().mean())
        print(f"  T{k:<8} {(f1_s > 0.5).sum():>6}개  "
              f"  {f1_s.max():>7.1%}   {f1_s.mean():>7.1%}   {proxy_acc:>8.1%}")

    # 클래스 커버리지
    coverage = (f1_mat > 0.3).sum(axis=0)
    print(f"\n  커버리지 (F1>0.3인 Teacher 수):")
    print(f"    전문가 0명: {int((coverage==0).sum())}개 클래스")
    print(f"    전문가 1명: {int((coverage==1).sum())}개 클래스")
    print(f"    전문가 2명+: {int((coverage>=2).sum())}개 클래스")

    # =====================================
    # Layer 2 지표 (logit 품질)
    # =====================================
    print(f"\n--- Logit Quality 지표 (Layer 2) ---")
    quality = {}
    for k in range(NUM_CLIENTS):
        q = logit_quality_metrics(
            teacher_logits[k], proxy_labels,
            expertise_f1=f1_mat[k], expert_threshold=0.5)
        quality[f"teacher_{k}"] = q
        print(f"  T{k}: entropy={q['mean_entropy']:.3f} "
              f"top1_conf={q['mean_top1_conf']:.3f} "
              f"top2_gap={q['mean_top2_gap']:.3f}", end="")
        if "expert_mean_entropy" in q:
            print(f" | expert_ent={q['expert_mean_entropy']:.3f} "
                  f"nonexp_ent={q.get('nonexpert_mean_entropy', 0):.3f}")
        else:
            print()

    # =====================================
    # 메타데이터 저장
    # =====================================
    meta = {
        "alpha": alpha,
        "seed": seed,
        "num_clients": NUM_CLIENTS,
        "num_classes": NUM_CLASSES,
        "epochs_teacher": EPOCHS_TEACHER,
        "teacher_val_metrics": teacher_val_metrics,
        "coverage_stats": {
            "n_uncovered_classes": int((coverage == 0).sum()),
            "n_single_expert_classes": int((coverage == 1).sum()),
            "n_multi_expert_classes": int((coverage >= 2).sum()),
        },
        "logit_quality": quality,
        "time_summaries": time_summaries,
    }
    save_json(meta, meta_path)

    print(f"\n===== α={alpha}, seed={seed} 완료 =====")
    print(f"  teachers.pt       → {teachers_path}")
    print(f"  teacher_logits.pt → {logits_path}")
    print(f"  expertise.npz     → {expertise_path}")
    print(f"  metadata.json     → {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    run_teachers(args.alpha, args.seed,
                 use_amp=not args.no_amp, force=args.force)


if __name__ == "__main__":
    main()
