"""
aggregate_sweep.py — 모든 실험 결과 집계 및 시각화
======================================================

모든 (α, seed, weighting) 조합의 결과를 읽어들여
보고서용 Figure와 요약 테이블을 생성한다.

생성되는 Figure (results_root/figures/ 에 저장):
  Figure 1a — Client×Class 데이터 할당 히트맵 (α별 subplot)
  Figure 1b — Client별 활성 클래스 수 (α에 따른 변화)
  Figure 1c — Client별 데이터 총량 stacked bar
  Figure 2a — Expertise Matrix 히트맵 (α별 subplot)
  Figure 2b — Teacher 성능 박스플롯 (α별)
  Figure 3a — Logit Entropy 분포 (violin, α별)
  Figure 3b — Top-1 Confidence CDF (α별)
  Figure 3c — 전문/비전문 클래스별 Entropy 비교
  Figure 4a — 핵심 역상관 그래프 (Teacher F1 vs Student Acc)
  Figure 4b — Weighting별 Student Acc (α 스윕)
  Figure 4c — Gap Recovery 곡선
  Figure 4d — Per-class accuracy 비교 (α=0.1 대표)

Summary table:
  summary_table.csv — 모든 조합의 정량 지표 한 장 요약

실행:
  python aggregate_sweep.py
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import torch

from common import (
    RESULTS_ROOT, ensure_dirs,
    partition_path, teachers_dir, kd_dir, bounds_dir,
    ParquetImageDataset, load_parquet_table,
)


ALPHAS = [0.1, 0.5, 1.0, 10.0, 100.0]
SEEDS = [42, 123]
WEIGHTINGS = ["uniform", "top_1", "top_3"]
NUM_CLIENTS = 5
NUM_CLASSES = 100

FIG_DIR = RESULTS_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 색상 팔레트
WEIGHT_COLORS = {
    "uniform": "#2563EB",  # 파랑
    "top_1":   "#DC2626",  # 빨강
    "top_3":   "#059669",  # 초록
}


# =============================================================================
# 유틸
# =============================================================================
def safe_load_json(p):
    p = Path(p)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def collect_all_kd_metrics():
    """모든 KD run 결과를 DataFrame으로."""
    rows = []
    for alpha in ALPHAS:
        for seed in SEEDS:
            for w in WEIGHTINGS:
                m = safe_load_json(kd_dir(alpha, seed, w) / "metrics.json")
                if m is None:
                    print(f"  MISSING: α={alpha}, seed={seed}, w={w}")
                    continue
                rows.append({
                    "alpha": alpha, "seed": seed, "weighting": w,
                    "student_acc": m["final_accuracy"],
                    "student_macro_f1": m["final_macro_f1"],
                    "gap_recovery": m.get("gap_recovery_pct"),
                    "lower_acc": m.get("lower_acc"),
                    "upper_acc": m.get("upper_acc"),
                    "best_epoch": m["best_epoch"],
                })
    return pd.DataFrame(rows)


def collect_teacher_metadata():
    """Teacher 관련 메타데이터 수집."""
    data = {}
    for alpha in ALPHAS:
        for seed in SEEDS:
            meta = safe_load_json(teachers_dir(alpha, seed) / "metadata.json")
            if meta is None:
                continue
            data[(alpha, seed)] = meta
    return data


def collect_bounds():
    """Lower/Upper Bound 값 (seed별)."""
    result = {}
    for seed in SEEDS:
        low = safe_load_json(bounds_dir(seed) / "lower_metrics.json")
        up = safe_load_json(bounds_dir(seed) / "upper_metrics.json")
        result[seed] = {
            "lower": low["final_accuracy"] if low else None,
            "upper": up["final_accuracy"] if up else None,
        }
    return result


# =============================================================================
# Figure 1 — 파티션 시각화
# =============================================================================
def fig1_partition_visualization(seed=42):
    """α별 client-class 분배 히트맵, 활성 클래스 수, 데이터 총량."""
    print("\n[Figure 1] 파티션 시각화 생성 중...")

    # 라벨 로드 (targets는 한 번만 읽으면 됨)
    # parquet에서 label 컬럼만 추출 — 이미지 바이트는 건드리지 않아 빠름
    import pyarrow.parquet as pq
    from common import TRAIN_PARQUET
    labels_all = np.array(
        pq.read_table(str(TRAIN_PARQUET), columns=["label"])
          .column("label").to_pylist()
    )

    # --- Fig 1a: α별 client-class 히트맵 ---
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(4 * len(ALPHAS), 4.5),
                             sharey=True)
    vmax = 0
    heat_data = []
    for alpha in ALPHAS:
        p = np.load(partition_path(alpha, seed))
        dist = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=int)
        for k in range(NUM_CLIENTS):
            idx = p[f"client_{k}"]
            cnt = np.bincount(labels_all[idx], minlength=NUM_CLASSES)
            dist[k] = cnt
        heat_data.append((alpha, dist))
        vmax = max(vmax, dist.max())

    for ax, (alpha, dist) in zip(axes, heat_data):
        # log scale for better visibility
        im = ax.imshow(np.log10(dist + 1), aspect="auto",
                       cmap="viridis", vmin=0, vmax=np.log10(vmax + 1))
        ax.set_title(f"α = {alpha}", fontsize=11)
        ax.set_xlabel("Class")
        if ax is axes[0]:
            ax.set_ylabel("Client")
        ax.set_yticks(range(NUM_CLIENTS))

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("log₁₀(count + 1)")
    fig.suptitle(f"Figure 1a — Client × Class data allocation (seed={seed})",
                 fontsize=12, y=1.02)
    plt.savefig(FIG_DIR / "fig1a_partition_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 1b: 활성 클래스 수 ---
    fig, ax = plt.subplots(figsize=(7, 4))
    means, stds = [], []
    for alpha, dist in heat_data:
        active_cnts = (dist > 0).sum(axis=1)  # K 명의 활성 클래스 수
        means.append(active_cnts.mean())
        stds.append(active_cnts.std())
    xs = np.arange(len(ALPHAS))
    ax.errorbar(xs, means, yerr=stds, fmt="o-", color="#2563EB",
                markersize=8, capsize=5, linewidth=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(a) for a in ALPHAS])
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Active classes per Teacher")
    ax.set_title("Figure 1b — Teacher coverage narrows as α decreases")
    ax.grid(alpha=0.3)
    ax.axhline(y=NUM_CLASSES, color="gray", linestyle="--",
               alpha=0.5, label=f"All {NUM_CLASSES} classes")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1b_active_classes.png", dpi=150)
    plt.close()

    # --- Fig 1c: Client별 데이터 총량 (α별) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.15
    xs = np.arange(NUM_CLIENTS)
    for i, (alpha, dist) in enumerate(heat_data):
        totals = dist.sum(axis=1)
        ax.bar(xs + i * width - 0.3, totals, width,
               label=f"α={alpha}", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"T{k}" for k in range(NUM_CLIENTS)])
    ax.set_xlabel("Teacher")
    ax.set_ylabel("Total samples")
    ax.set_title("Figure 1c — Total samples per Teacher (sanity check)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1c_total_samples.png", dpi=150)
    plt.close()
    print("  saved Figures 1a, 1b, 1c")


# =============================================================================
# Figure 2 — Teacher 성능 시각화
# =============================================================================
def fig2_teacher_performance(seed=42):
    """Expertise Matrix 히트맵 및 Teacher 성능 박스플롯."""
    print("\n[Figure 2] Teacher 성능 시각화 생성 중...")

    # Fig 2a: Expertise Matrix α별 히트맵
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(4 * len(ALPHAS), 3.8),
                             sharey=True)
    for ax, alpha in zip(axes, ALPHAS):
        exp_path = teachers_dir(alpha, seed) / "expertise.npz"
        if not exp_path.exists():
            ax.set_title(f"α={alpha}\n(no data)")
            continue
        f1 = np.load(exp_path)["f1"]  # (K, C)
        im = ax.imshow(f1, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=1)
        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Class")
        if ax is axes[0]:
            ax.set_ylabel("Teacher")
        ax.set_yticks(range(NUM_CLIENTS))
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("per-class F1")
    fig.suptitle(f"Figure 2a — Expertise Matrix across α (seed={seed})",
                 fontsize=12, y=1.02)
    plt.savefig(FIG_DIR / "fig2a_expertise_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2b: Teacher 성능 박스플롯 (seed 집계)
    fig, ax = plt.subplots(figsize=(8, 5))
    box_data = []
    for alpha in ALPHAS:
        accs = []
        for s in SEEDS:
            meta = safe_load_json(teachers_dir(alpha, s) / "metadata.json")
            if meta is None:
                continue
            for k in range(NUM_CLIENTS):
                acc = meta["teacher_val_metrics"][str(k)]["val_accuracy"]
                accs.append(acc)
        box_data.append(accs)
    bp = ax.boxplot(box_data, labels=[str(a) for a in ALPHAS],
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#93C5FD")
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Teacher Val Accuracy")
    ax.set_title("Figure 2b — Individual Teacher performance (val)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2b_teacher_boxplot.png", dpi=150)
    plt.close()
    print("  saved Figures 2a, 2b")


# =============================================================================
# Figure 3 — Logit Quality (Layer 2)
# =============================================================================
def fig3_logit_quality():
    """Logit entropy, top-1 confidence, 전문/비전문 비교."""
    print("\n[Figure 3] Logit quality 시각화 생성 중...")

    # 지표 수집
    entropy_by_alpha = defaultdict(list)        # alpha -> [sample entropies...]
    top1_by_alpha = defaultdict(list)
    expert_ent_by_alpha = defaultdict(list)     # (alpha, seed) 평균값 모음
    nonexpert_ent_by_alpha = defaultdict(list)

    for alpha in ALPHAS:
        for seed in SEEDS:
            meta = safe_load_json(teachers_dir(alpha, seed) / "metadata.json")
            if meta is None:
                continue
            for k in range(NUM_CLIENTS):
                q = meta["logit_quality"][f"teacher_{k}"]
                entropy_by_alpha[alpha].extend(q["entropy_values"])
                top1_by_alpha[alpha].extend(q["top1_conf_values"])
                if "expert_mean_entropy" in q:
                    expert_ent_by_alpha[alpha].append(q["expert_mean_entropy"])
                if "nonexpert_mean_entropy" in q:
                    nonexpert_ent_by_alpha[alpha].append(q["nonexpert_mean_entropy"])

    # Fig 3a: entropy violin
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [entropy_by_alpha[a] for a in ALPHAS if entropy_by_alpha[a]]
    if data:
        parts = ax.violinplot(data, positions=range(len(data)),
                              showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#FBBF24")
            pc.set_alpha(0.6)
    ax.set_xticks(range(len(ALPHAS)))
    ax.set_xticklabels([str(a) for a in ALPHAS])
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Logit entropy (nat)")
    ax.set_title("Figure 3a — Logit entropy distribution across α\n"
                 "(low α → low entropy → hard-label-like Teacher logits)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3a_entropy_violin.png", dpi=150)
    plt.close()

    # Fig 3b: top-1 confidence CDF
    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha in ALPHAS:
        vals = sorted(top1_by_alpha[alpha])
        if not vals:
            continue
        ys = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, ys, label=f"α={alpha}", linewidth=2)
    ax.set_xlabel("Top-1 softmax confidence")
    ax.set_ylabel("CDF (fraction of samples)")
    ax.set_title("Figure 3b — Top-1 confidence CDF\n"
                 "(low α → curve shifts right → confidence concentrates near 1)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3b_top1_cdf.png", dpi=150)
    plt.close()

    # Fig 3c: 전문/비전문 비교
    fig, ax = plt.subplots(figsize=(8, 5))
    exp_means, exp_stds = [], []
    nonexp_means, nonexp_stds = [], []
    xs = np.arange(len(ALPHAS))
    for alpha in ALPHAS:
        if expert_ent_by_alpha[alpha]:
            exp_means.append(np.mean(expert_ent_by_alpha[alpha]))
            exp_stds.append(np.std(expert_ent_by_alpha[alpha]))
        else:
            exp_means.append(np.nan); exp_stds.append(0)
        if nonexpert_ent_by_alpha[alpha]:
            nonexp_means.append(np.mean(nonexpert_ent_by_alpha[alpha]))
            nonexp_stds.append(np.std(nonexpert_ent_by_alpha[alpha]))
        else:
            nonexp_means.append(np.nan); nonexp_stds.append(0)

    ax.errorbar(xs - 0.05, exp_means, yerr=exp_stds, fmt="o-",
                label="Expert classes (F1>0.5)", color="#059669",
                markersize=8, linewidth=2, capsize=4)
    ax.errorbar(xs + 0.05, nonexp_means, yerr=nonexp_stds, fmt="s-",
                label="Non-expert classes (F1≤0.5)", color="#DC2626",
                markersize=8, linewidth=2, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(a) for a in ALPHAS])
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Mean logit entropy")
    ax.set_title("Figure 3c — Entropy: expert vs non-expert classes\n"
                 "(dark knowledge absent in both regimes at low α)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3c_expert_vs_nonexpert_entropy.png", dpi=150)
    plt.close()
    print("  saved Figures 3a, 3b, 3c")


# =============================================================================
# Figure 4 — Student results & main claim
# =============================================================================
def fig4_student_results(df: pd.DataFrame, bounds_map: dict):
    """핵심 결과 그래프들."""
    print("\n[Figure 4] Student 결과 시각화 생성 중...")

    # 평균 bounds
    lower_mean = np.nanmean([bounds_map[s]["lower"] for s in SEEDS
                             if bounds_map[s]["lower"] is not None])
    upper_mean = np.nanmean([bounds_map[s]["upper"] for s in SEEDS
                             if bounds_map[s]["upper"] is not None])

    # --- Fig 4a: 핵심 역상관 ---
    # x: α (log), y1: Teacher F1 평균 (전문 클래스), y2: Student Acc (uniform 대표)
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    # Teacher F1: α별 평균 (전문 클래스만, F1>0.5의 평균)
    teacher_f1_means, teacher_f1_stds = [], []
    for alpha in ALPHAS:
        vals = []
        for seed in SEEDS:
            exp_path = teachers_dir(alpha, seed) / "expertise.npz"
            if not exp_path.exists():
                continue
            f1 = np.load(exp_path)["f1"]
            # 각 Teacher별 전문 클래스 평균 F1
            for k in range(NUM_CLIENTS):
                expert_f1 = f1[k][f1[k] > 0.5]
                if len(expert_f1) > 0:
                    vals.append(expert_f1.mean())
        teacher_f1_means.append(np.mean(vals) if vals else np.nan)
        teacher_f1_stds.append(np.std(vals) if vals else 0)

    xs = np.arange(len(ALPHAS))
    ax1.errorbar(xs, teacher_f1_means, yerr=teacher_f1_stds,
                 fmt="o-", color="#F59E0B", markersize=10,
                 linewidth=2.5, capsize=5,
                 label="Teacher F1 (expert classes)")
    ax1.set_xlabel("Dirichlet α")
    ax1.set_ylabel("Teacher per-class F1 (expert classes only)", color="#F59E0B")
    ax1.tick_params(axis="y", labelcolor="#F59E0B")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(a) for a in ALPHAS])
    ax1.set_ylim(0, 1.0)
    ax1.grid(alpha=0.3)

    # Student Acc (uniform weighting, seed 평균)
    ax2 = ax1.twinx()
    student_means, student_stds = [], []
    for alpha in ALPHAS:
        sub = df[(df["alpha"] == alpha) & (df["weighting"] == "uniform")]
        student_means.append(sub["student_acc"].mean() if len(sub) else np.nan)
        student_stds.append(sub["student_acc"].std() if len(sub) else 0)

    ax2.errorbar(xs, student_means, yerr=student_stds,
                 fmt="s-", color="#2563EB", markersize=10,
                 linewidth=2.5, capsize=5,
                 label="Student Acc (Uniform KD)")
    ax2.set_ylabel("Student global accuracy", color="#2563EB")
    ax2.tick_params(axis="y", labelcolor="#2563EB")

    # bounds 수평선
    ax2.axhline(y=lower_mean, color="gray", linestyle="--",
                alpha=0.6, label=f"Lower Bound ({lower_mean:.3f})")
    ax2.axhline(y=upper_mean, color="black", linestyle="--",
                alpha=0.6, label=f"Upper Bound ({upper_mean:.3f})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    plt.title("Figure 4a — Main claim: Teacher F1 vs Student Acc (inverse correlation)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4a_main_inverse_correlation.png", dpi=150)
    plt.close()

    # --- Fig 4b: Weighting별 Student Acc ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for w in WEIGHTINGS:
        means, stds = [], []
        for alpha in ALPHAS:
            sub = df[(df["alpha"] == alpha) & (df["weighting"] == w)]
            means.append(sub["student_acc"].mean() if len(sub) else np.nan)
            stds.append(sub["student_acc"].std() if len(sub) else 0)
        ax.errorbar(xs, means, yerr=stds, fmt="o-",
                    color=WEIGHT_COLORS[w], markersize=10, linewidth=2.5,
                    capsize=5, label=w)

    ax.axhline(y=lower_mean, color="gray", linestyle="--",
               alpha=0.6, label=f"Lower Bound ({lower_mean:.3f})")
    ax.axhline(y=upper_mean, color="black", linestyle="--",
               alpha=0.6, label=f"Upper Bound ({upper_mean:.3f})")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(a) for a in ALPHAS])
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Student global accuracy")
    ax.set_title("Figure 4b — Student accuracy by weighting method")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4b_weighting_comparison.png", dpi=150)
    plt.close()

    # --- Fig 4c: Gap Recovery ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for w in WEIGHTINGS:
        means, stds = [], []
        for alpha in ALPHAS:
            sub = df[(df["alpha"] == alpha) & (df["weighting"] == w)]
            vals = sub["gap_recovery"].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std() if len(vals) else 0)
        ax.errorbar(xs, means, yerr=stds, fmt="o-",
                    color=WEIGHT_COLORS[w], markersize=10, linewidth=2.5,
                    capsize=5, label=w)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Lower Bound (0%)")
    ax.axhline(y=100, color="black", linestyle="--", alpha=0.5, label="Upper Bound (100%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(a) for a in ALPHAS])
    ax.set_xlabel("Dirichlet α")
    ax.set_ylabel("Gap Recovery (%)")
    ax.set_title("Figure 4c — Gap Recovery: negative = worse than Lower Bound")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4c_gap_recovery.png", dpi=150)
    plt.close()

    # --- Fig 4d: Per-class acc (α=0.1 대표) ---
    fig, ax = plt.subplots(figsize=(14, 5))
    # 기준 정렬: uniform의 per-class acc 순서
    ref_metrics = safe_load_json(kd_dir(0.1, SEEDS[0], "uniform") / "metrics.json")
    if ref_metrics is not None:
        ref_accs = np.array([
            p["recall"]  # per-class accuracy = recall (true positive rate)
            for p in sorted(ref_metrics["per_class"], key=lambda x: x["class"])
        ])
        order = np.argsort(ref_accs)

        for w in WEIGHTINGS:
            m = safe_load_json(kd_dir(0.1, SEEDS[0], w) / "metrics.json")
            if m is None:
                continue
            accs = np.array([
                p["recall"] for p in sorted(m["per_class"], key=lambda x: x["class"])
            ])
            ax.plot(range(NUM_CLASSES), accs[order], "o-",
                    color=WEIGHT_COLORS[w], markersize=3, linewidth=0.8,
                    alpha=0.7, label=w)
        ax.set_xlabel("Class (sorted by Uniform recall)")
        ax.set_ylabel("Per-class recall")
        ax.set_title(f"Figure 4d — Per-class recall comparison (α=0.1, seed={SEEDS[0]})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig4d_per_class_alpha01.png", dpi=150)
    plt.close()
    print("  saved Figures 4a, 4b, 4c, 4d")


# =============================================================================
# Summary Table
# =============================================================================
def save_summary_table(df: pd.DataFrame, bounds_map: dict):
    """정량 지표를 CSV로."""
    # raw
    raw_path = FIG_DIR / "summary_raw.csv"
    df.to_csv(raw_path, index=False, float_format="%.4f")

    # aggregated (seed 평균)
    agg = df.groupby(["alpha", "weighting"]).agg(
        student_acc_mean=("student_acc", "mean"),
        student_acc_std=("student_acc", "std"),
        gap_recovery_mean=("gap_recovery", "mean"),
        gap_recovery_std=("gap_recovery", "std"),
        macro_f1_mean=("student_macro_f1", "mean"),
    ).reset_index()
    agg_path = FIG_DIR / "summary_aggregated.csv"
    agg.to_csv(agg_path, index=False, float_format="%.4f")

    print(f"\n[Summary] saved:")
    print(f"  raw        → {raw_path}")
    print(f"  aggregated → {agg_path}")
    print(f"\n=== Aggregated Summary ===")
    print(agg.to_string(index=False))


# =============================================================================
# 메인
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-seed", type=int, default=42,
                        help="파티션 시각화용 대표 seed")
    parser.add_argument("--skip-figs", action="store_true",
                        help="Figure 생성 건너뜀 (데이터 집계만)")
    args = parser.parse_args()

    ensure_dirs()

    # 데이터 수집
    df = collect_all_kd_metrics()
    bounds_map = collect_bounds()

    print(f"\n=== 결과 수집 요약 ===")
    print(f"  Total KD runs: {len(df)}")
    print(f"  Bounds: {bounds_map}")

    if df.empty:
        print("  수집된 데이터가 없습니다. 먼저 실험을 돌리세요.")
        return

    save_summary_table(df, bounds_map)

    if not args.skip_figs:
        fig1_partition_visualization(seed=args.partition_seed)
        fig2_teacher_performance(seed=args.partition_seed)
        fig3_logit_quality()
        fig4_student_results(df, bounds_map)

    print(f"\n=== 모든 Figure는 {FIG_DIR} 에 저장되었습니다 ===")


if __name__ == "__main__":
    main()
