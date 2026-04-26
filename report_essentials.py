"""
report_essentials.py — 보고서/논문용 핵심 결과만 추출
=========================================================

`aggregate_sweep.py`가 생성한 12개 Figure 중에서
보고서 본문에 들어갈 만한 핵심만 추려서 별도 디렉토리로 복사하고,
본문용 정량 표(Markdown + CSV)를 함께 생성한다.

생성물:
  $EACS_RESULTS_ROOT/report_essentials/
  ├── phase1_main/                    Phase 1 메인 Figure (주장 A)
  │   ├── fig3a_entropy_violin.png
  │   ├── fig3b_top1_cdf.png
  │   ├── fig3c_expert_vs_nonexpert.png
  │   └── fig4b_student_by_weighting.png
  ├── phase2_main/                    Phase 2 메인 Figure (주장 B)
  │   ├── fig4b_kd_below_lower.png
  │   └── fig4c_gap_recovery_negative.png
  ├── appendix/                       부록용 Figure
  │   ├── phase1_partition.png
  │   └── phase1_expertise_matrix.png
  ├── tables/                         정량 표
  │   ├── table_phase1_summary.md
  │   ├── table_phase1_summary.csv
  │   ├── table_phase2_summary.md
  │   ├── table_phase2_summary.csv
  │   └── table_combined_overview.md  ← 한 페이지 요약
  └── REPORT_GUIDE.md                  보고서에 삽입할 때 가이드

실행:
  python report_essentials.py
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from common import (
    RESULTS_ROOT, ensure_dirs,
    teachers_dir, kd_dir, bounds_dir,
)


ALPHAS_PHASE1 = [0.1, 0.5, 1.0, 10.0, 100.0]
ALPHAS_PHASE2 = [0.1, 1.0, 100.0]
SEEDS = [42, 123]
WEIGHTINGS = ["uniform", "top_1", "top_3"]
NUM_CLIENTS = 5
NUM_CLASSES = 100

OUT_ROOT = RESULTS_ROOT / "report_essentials"


# =============================================================================
# 유틸
# =============================================================================
def safe_load_json(p):
    p = Path(p)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fig_dir_for(phase):
    return RESULTS_ROOT / f"phase{phase}" / "figures"


def copy_figure(src, dst, label=None):
    """소스 Figure를 대상 위치로 복사. 없으면 경고만."""
    src = Path(src); dst = Path(dst)
    if not src.exists():
        print(f"  [MISSING] {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied: {label or src.name} → {dst}")
    return True


# =============================================================================
# Phase 1 정량 표 생성 (주장 A)
# =============================================================================
def build_phase1_table():
    """Phase 1: α별로 Teacher logit 품질 + Student acc + Gap Recovery."""
    rows = []

    for alpha in ALPHAS_PHASE1:
        # Teacher logit quality (Layer 2)
        entropies, top1s = [], []
        for seed in SEEDS:
            meta = safe_load_json(teachers_dir(alpha, seed, phase=1) / "metadata.json")
            if meta is None:
                continue
            for k in range(NUM_CLIENTS):
                q = meta["logit_quality"][f"teacher_{k}"]
                entropies.append(q["mean_entropy"])
                top1s.append(q["mean_top1_conf"])

        # Student acc by weighting (Layer 3)
        student_accs = {}
        gap_recoveries = {}
        for w in WEIGHTINGS:
            accs, gaps = [], []
            for seed in SEEDS:
                m = safe_load_json(kd_dir(alpha, seed, w, phase=1) / "metrics.json")
                if m is None:
                    continue
                accs.append(m["final_accuracy"])
                if m.get("gap_recovery_pct") is not None:
                    gaps.append(m["gap_recovery_pct"])
            student_accs[w] = (np.mean(accs) if accs else np.nan,
                               np.std(accs) if accs else 0)
            gap_recoveries[w] = (np.mean(gaps) if gaps else np.nan,
                                 np.std(gaps) if gaps else 0)

        rows.append({
            "alpha": alpha,
            "teacher_mean_entropy": np.mean(entropies) if entropies else np.nan,
            "teacher_mean_top1_conf": np.mean(top1s) if top1s else np.nan,
            "uniform_acc_mean": student_accs["uniform"][0],
            "uniform_acc_std": student_accs["uniform"][1],
            "top_1_acc_mean": student_accs["top_1"][0],
            "top_3_acc_mean": student_accs["top_3"][0],
            "uniform_gap_recovery": gap_recoveries["uniform"][0],
            "top_1_gap_recovery": gap_recoveries["top_1"][0],
            "top_3_gap_recovery": gap_recoveries["top_3"][0],
        })

    df = pd.DataFrame(rows)

    # Lower / Upper Bound (α 무관)
    lowers, uppers = [], []
    for seed in SEEDS:
        low = safe_load_json(bounds_dir(seed, phase=1) / "lower_metrics.json")
        up = safe_load_json(bounds_dir(seed, phase=1) / "upper_metrics.json")
        if low: lowers.append(low["final_accuracy"])
        if up: uppers.append(up["final_accuracy"])
    bounds_info = {
        "lower": np.mean(lowers) if lowers else np.nan,
        "upper": np.mean(uppers) if uppers else np.nan,
    }
    return df, bounds_info


# =============================================================================
# Phase 2 정량 표 생성 (주장 B)
# =============================================================================
def build_phase2_table():
    rows = []
    for alpha in ALPHAS_PHASE2:
        student_accs = {}
        gap_recoveries = {}
        for w in WEIGHTINGS:
            accs, gaps = [], []
            for seed in SEEDS:
                m = safe_load_json(kd_dir(alpha, seed, w, phase=2) / "metrics.json")
                if m is None:
                    continue
                accs.append(m["final_accuracy"])
                if m.get("gap_recovery_pct") is not None:
                    gaps.append(m["gap_recovery_pct"])
            student_accs[w] = (np.mean(accs) if accs else np.nan,
                               np.std(accs) if accs else 0)
            gap_recoveries[w] = (np.mean(gaps) if gaps else np.nan,
                                 np.std(gaps) if gaps else 0)

        rows.append({
            "alpha": alpha,
            "uniform_acc_mean": student_accs["uniform"][0],
            "uniform_acc_std": student_accs["uniform"][1],
            "top_1_acc_mean": student_accs["top_1"][0],
            "top_3_acc_mean": student_accs["top_3"][0],
            "uniform_gap_recovery": gap_recoveries["uniform"][0],
            "top_1_gap_recovery": gap_recoveries["top_1"][0],
            "top_3_gap_recovery": gap_recoveries["top_3"][0],
        })

    df = pd.DataFrame(rows)
    lowers, uppers = [], []
    for seed in SEEDS:
        low = safe_load_json(bounds_dir(seed, phase=2) / "lower_metrics.json")
        up = safe_load_json(bounds_dir(seed, phase=2) / "upper_metrics.json")
        if low: lowers.append(low["final_accuracy"])
        if up: uppers.append(up["final_accuracy"])
    bounds_info = {
        "lower": np.mean(lowers) if lowers else np.nan,
        "upper": np.mean(uppers) if uppers else np.nan,
    }
    return df, bounds_info


# =============================================================================
# Markdown 표 생성
# =============================================================================
def df_to_markdown_phase1(df, bounds):
    if df.empty:
        return "_(데이터 없음)_"

    lines = []
    lines.append("**Phase 1 — 주장 A (동일 ResNet-18 pretrained)**\n")
    lines.append(f"- Lower Bound: **{bounds['lower']:.4f}** (proxy만)")
    lines.append(f"- Upper Bound: **{bounds['upper']:.4f}** (centralized)\n")

    # 헤더
    lines.append("| α | T entropy↓ | T top-1↑ | Uniform | Top-1 | Top-3 | Uniform GR | Top-1 GR | Top-3 GR |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for _, r in df.iterrows():
        lines.append(
            f"| {r['alpha']} "
            f"| {r['teacher_mean_entropy']:.3f} "
            f"| {r['teacher_mean_top1_conf']:.3f} "
            f"| {r['uniform_acc_mean']:.4f} "
            f"| {r['top_1_acc_mean']:.4f} "
            f"| {r['top_3_acc_mean']:.4f} "
            f"| {r['uniform_gap_recovery']:+.1f}% "
            f"| {r['top_1_gap_recovery']:+.1f}% "
            f"| {r['top_3_gap_recovery']:+.1f}% |"
        )

    lines.append("")
    lines.append("**해석**:")
    lines.append("- T entropy: Teacher logit의 평균 entropy. α↓일수록 ↓이면 hard-label화 진행")
    lines.append("- T top-1: top-1 softmax 평균. α↓일수록 ↑이면 dark knowledge 부재")
    lines.append("- GR (Gap Recovery): 0%=Lower, 100%=Upper, 음수=Lower보다 나쁨")
    return "\n".join(lines)


def df_to_markdown_phase2(df, bounds):
    if df.empty:
        return "_(데이터 없음)_"

    lines = []
    lines.append("**Phase 2 — 주장 B (Small→Large: MobileNetV2 → ResNet-50, pretrained)**\n")
    lines.append(f"- Lower Bound: **{bounds['lower']:.4f}** (proxy만)")
    lines.append(f"- Upper Bound: **{bounds['upper']:.4f}** (centralized)\n")

    lines.append("| α | Uniform | Top-1 | Top-3 | Uniform GR | Top-1 GR | Top-3 GR |")
    lines.append("|---|---|---|---|---|---|---|")

    for _, r in df.iterrows():
        lines.append(
            f"| {r['alpha']} "
            f"| {r['uniform_acc_mean']:.4f} "
            f"| {r['top_1_acc_mean']:.4f} "
            f"| {r['top_3_acc_mean']:.4f} "
            f"| {r['uniform_gap_recovery']:+.1f}% "
            f"| {r['top_1_gap_recovery']:+.1f}% "
            f"| {r['top_3_gap_recovery']:+.1f}% |"
        )

    lines.append("")
    lines.append("**해석**:")
    lines.append("- Lower가 Upper에 매우 근접 (작은 Gap) → Student가 이미 강함")
    lines.append("- α=0.1에서 Uniform/Top-1/Top-3가 Lower 아래로 떨어지면 핵심 증거")
    lines.append("- Gap Recovery 음수 = Non-IID Teacher의 KD가 Student를 끌어내림")
    return "\n".join(lines)


# =============================================================================
# Figure 큐레이션
# =============================================================================
def curate_phase1_figures():
    """Phase 1 핵심 4개 + 부록 2개."""
    src = fig_dir_for(1)
    dst_main = OUT_ROOT / "phase1_main"
    dst_appx = OUT_ROOT / "appendix"

    print("\n[Phase 1 메인 Figure 큐레이션]")
    copy_figure(src / "fig3a_entropy_violin.png",
                dst_main / "fig3a_entropy_violin.png",
                "Fig 3a — Logit entropy 분포 (α별)")
    copy_figure(src / "fig3b_top1_cdf.png",
                dst_main / "fig3b_top1_cdf.png",
                "Fig 3b — Top-1 confidence CDF")
    copy_figure(src / "fig3c_expert_vs_nonexpert_entropy.png",
                dst_main / "fig3c_expert_vs_nonexpert.png",
                "Fig 3c — 전문/비전문 클래스 entropy")
    copy_figure(src / "fig4b_weighting_comparison.png",
                dst_main / "fig4b_student_by_weighting.png",
                "Fig 4b — Student acc by weighting")

    print("\n[Phase 1 부록 Figure]")
    copy_figure(src / "fig1a_partition_heatmap.png",
                dst_appx / "phase1_partition.png",
                "Phase 1 파티션")
    copy_figure(src / "fig2a_expertise_heatmap.png",
                dst_appx / "phase1_expertise_matrix.png",
                "Phase 1 Expertise Matrix")


def curate_phase2_figures():
    """Phase 2 핵심 2개."""
    src = fig_dir_for(2)
    dst_main = OUT_ROOT / "phase2_main"

    print("\n[Phase 2 메인 Figure 큐레이션]")
    copy_figure(src / "fig4b_weighting_comparison.png",
                dst_main / "fig4b_kd_below_lower.png",
                "Fig 4b — KD vs Lower Bound")
    copy_figure(src / "fig4c_gap_recovery.png",
                dst_main / "fig4c_gap_recovery_negative.png",
                "Fig 4c — Gap Recovery (음수 영역)")


# =============================================================================
# 통합 보고서 가이드
# =============================================================================
def write_report_guide(p1_df, p1_bounds, p2_df, p2_bounds):
    guide_path = OUT_ROOT / "REPORT_GUIDE.md"

    lines = []
    lines.append("# 보고서/논문 작성 가이드 — 핵심 정수")
    lines.append("")
    lines.append("이 디렉토리는 보고서 본문에 들어갈 **핵심 Figure 6개**와 "
                 "**정량 표 2개**를 포함합니다.")
    lines.append("")
    lines.append("## 권장 보고서 구조")
    lines.append("")
    lines.append("### Section 1 — 논지 요약")
    lines.append("")
    lines.append('> "Non-IID Teacher의 logit은 dark knowledge가 부재하면 noise로 작용한다. '
                 '이 원리는 (i) 동일 모델 환경(Phase 1)에서 직접 측정 가능하며, '
                 '(ii) Small→Large 현실 시나리오(Phase 2)에서 KD 실패라는 형태로 재현된다."')
    lines.append("")
    lines.append("### Section 2 — Phase 1 (주장 A 본문)")
    lines.append("")
    lines.append("**필요 Figure**: `phase1_main/` 4개")
    lines.append("- Fig 3a, 3b, 3c: Layer 2 지표로 logit이 noise임을 직접 보임")
    lines.append("- Fig 4b: Student acc로 Layer 3 보조 증거")
    lines.append("")
    lines.append("**필요 표**: `tables/table_phase1_summary.md`")
    lines.append("")
    lines.append(df_to_markdown_phase1(p1_df, p1_bounds))
    lines.append("")
    lines.append("### Section 3 — Phase 2 (주장 B 본문)")
    lines.append("")
    lines.append("**필요 Figure**: `phase2_main/` 2개")
    lines.append("- Fig 4b: KD가 Lower 아래로 떨어지는 시각적 증거")
    lines.append("- Fig 4c: Gap Recovery 음수 진입")
    lines.append("")
    lines.append("**필요 표**: `tables/table_phase2_summary.md`")
    lines.append("")
    lines.append(df_to_markdown_phase2(p2_df, p2_bounds))
    lines.append("")
    lines.append("### Section 4 — 부록 / 보조 자료")
    lines.append("")
    lines.append("- `appendix/phase1_partition.png`: 실험 조건(Dirichlet 파티션) 시각화")
    lines.append("- `appendix/phase1_expertise_matrix.png`: Teacher 전문성 분포")
    lines.append("- 전체 12개 Figure는 `$EACS_RESULTS_ROOT/phase{1,2}/figures/`에 보관")
    lines.append("")
    lines.append("## 핵심 수치 한 줄 요약 (Abstract용)")
    lines.append("")

    if not p1_df.empty:
        # Phase 1 핵심 수치
        p1_low_alpha = p1_df[p1_df["alpha"] == 0.1].iloc[0] if (p1_df["alpha"] == 0.1).any() else None
        p1_high_alpha = p1_df[p1_df["alpha"] == 100.0].iloc[0] if (p1_df["alpha"] == 100.0).any() else None
        if p1_low_alpha is not None and p1_high_alpha is not None:
            lines.append(f"- **Phase 1**: α=0.1에서 Teacher logit entropy "
                         f"{p1_low_alpha['teacher_mean_entropy']:.2f} (vs IID α=100: "
                         f"{p1_high_alpha['teacher_mean_entropy']:.2f}) — "
                         f"hard-label화 명확히 관측")

    if not p2_df.empty:
        p2_low = p2_df[p2_df["alpha"] == 0.1].iloc[0] if (p2_df["alpha"] == 0.1).any() else None
        if p2_low is not None:
            lines.append(f"- **Phase 2**: α=0.1, Uniform KD = {p2_low['uniform_acc_mean']:.4f} "
                         f"(Lower {p2_bounds['lower']:.4f}와 비교)")
            lines.append(f"  Gap Recovery: Uniform {p2_low['uniform_gap_recovery']:+.1f}%, "
                         f"Top-1 {p2_low['top_1_gap_recovery']:+.1f}%")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 보고서에 직접 인용 가능한 핵심 문장 후보")
    lines.append("")

    if not p1_df.empty and not p2_df.empty:
        p1_low = p1_df[p1_df["alpha"] == 0.1].iloc[0] if (p1_df["alpha"] == 0.1).any() else None
        p1_high = p1_df[p1_df["alpha"] == 100.0].iloc[0] if (p1_df["alpha"] == 100.0).any() else None
        p2_low = p2_df[p2_df["alpha"] == 0.1].iloc[0] if (p2_df["alpha"] == 0.1).any() else None

        if p1_low is not None and p1_high is not None:
            ent_drop = p1_high['teacher_mean_entropy'] - p1_low['teacher_mean_entropy']
            lines.append(f'1. (Phase 1) "α=100(IID)에서 α=0.1로 갈수록 Teacher의 평균 logit '
                         f'entropy는 {p1_high["teacher_mean_entropy"]:.2f}에서 '
                         f'{p1_low["teacher_mean_entropy"]:.2f}로 {ent_drop:.2f} 감소했다 — '
                         f'dark knowledge가 사라지고 logit이 hard-label에 수렴함을 의미한다."')

        if p2_low is not None:
            lines.append(f'2. (Phase 2) "Pretrained Student는 proxy만으로 '
                         f'{p2_bounds["lower"]:.4f}의 Lower Bound를 달성하지만, '
                         f'α=0.1의 Non-IID Teacher KD는 {p2_low["uniform_acc_mean"]:.4f}로 '
                         f'**Lower Bound 아래**로 떨어진다 (Gap Recovery '
                         f'{p2_low["uniform_gap_recovery"]:+.1f}%)."')

        lines.append('3. (종합) "동일 모델(Phase 1)이든 Small→Large(Phase 2)이든, '
                     'α↓일수록 KD의 가치가 사라지는 일관된 패턴이 관측된다. '
                     '이는 logit 기반 KD 패러다임의 구조적 한계를 시사한다."')

    with open(guide_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  written: {guide_path}")


# =============================================================================
# 메인
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("보고서용 핵심 정수 추출")
    print("="*60)

    # --- 정량 표 생성 ---
    print("\n[정량 표 생성]")
    p1_df, p1_bounds = build_phase1_table()
    p2_df, p2_bounds = build_phase2_table()

    if not p1_df.empty:
        # CSV
        p1_df.to_csv(OUT_ROOT / "tables" / "table_phase1_summary.csv",
                     index=False, float_format="%.4f")
        # Markdown
        with open(OUT_ROOT / "tables" / "table_phase1_summary.md", "w") as f:
            f.write(df_to_markdown_phase1(p1_df, p1_bounds))
        print(f"  Phase 1: {len(p1_df)}개 α 행 생성")
    else:
        print("  Phase 1: 데이터 없음 (실험 미실행?)")

    if not p2_df.empty:
        p2_df.to_csv(OUT_ROOT / "tables" / "table_phase2_summary.csv",
                     index=False, float_format="%.4f")
        with open(OUT_ROOT / "tables" / "table_phase2_summary.md", "w") as f:
            f.write(df_to_markdown_phase2(p2_df, p2_bounds))
        print(f"  Phase 2: {len(p2_df)}개 α 행 생성")
    else:
        print("  Phase 2: 데이터 없음 (실험 미실행?)")

    # --- Figure 큐레이션 ---
    curate_phase1_figures()
    curate_phase2_figures()

    # --- 통합 가이드 ---
    write_report_guide(p1_df, p1_bounds, p2_df, p2_bounds)

    print("\n" + "="*60)
    print(f"완료. 결과는 다음 위치에 있습니다:")
    print(f"  {OUT_ROOT}")
    print("="*60)
    print("\n다음 파일을 확인하세요:")
    print(f"  - REPORT_GUIDE.md         보고서 작성 가이드")
    print(f"  - phase1_main/            Phase 1 메인 Figure 4개")
    print(f"  - phase2_main/            Phase 2 메인 Figure 2개")
    print(f"  - appendix/               부록 Figure 2개")
    print(f"  - tables/                 본문용 표 (md + csv)")


if __name__ == "__main__":
    main()
