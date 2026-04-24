"""
prepare_partition.py — 파티션 생성 스크립트
==============================================

목적:
  (α, seed) 조합별로 proxy indices + client private indices를 생성하여 .npz로 저장.
  모든 후속 스크립트(bounds/teachers/kd)가 이 파티션을 읽어 들어간다.

기본 실행:
  python prepare_partition.py --all   # 모든 α × seed 조합 생성

부분 실행 (EC2 분산 시 불필요, 수 분 내 끝남):
  python prepare_partition.py --alpha 0.1 --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    RESULTS_ROOT, DATA_ROOT, ensure_dirs,
    partition_path, set_seed,
    ParquetImageDataset, load_parquet_table, dirichlet_partition,
)


# =============================================================================
# 설정
# =============================================================================
ALPHAS = [0.1, 0.5, 1.0, 10.0, 100.0]
SEEDS = [42, 123]
NUM_CLASSES = 100
NUM_CLIENTS = 5
PROXY_PER_CLASS = 100


# =============================================================================
# 단일 파티션 생성
# =============================================================================
def build_partition(alpha: float, seed: int, dataset,
                    num_classes=NUM_CLASSES, num_clients=NUM_CLIENTS,
                    proxy_per_class=PROXY_PER_CLASS, force=False):
    """단일 (α, seed) 파티션 생성."""
    out_path = partition_path(alpha, seed)
    if out_path.exists() and not force:
        print(f"  SKIP (exists): {out_path}")
        return

    labels = np.array(dataset.targets)
    N = len(labels)

    rng = np.random.RandomState(seed)

    # 1) proxy 인덱스 추출 — 클래스별 균등
    proxy_indices = []
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proxy_indices.extend(idx_c[:proxy_per_class].tolist())
    proxy_indices = np.array(sorted(proxy_indices))
    proxy_set = set(proxy_indices.tolist())

    # 2) private pool — proxy 제외 전체
    private_indices = np.array([i for i in range(N) if i not in proxy_set])
    private_labels = labels[private_indices]

    # 3) Dirichlet 파티션 (private_indices 내 local index 기준)
    partition_local = dirichlet_partition(
        private_labels, num_clients, alpha, num_classes, seed)

    # local → global index 변환
    client_indices = {
        k: private_indices[partition_local[k]]
        for k in range(num_clients)
    }

    # 4) 통계 출력
    print(f"\n=== α={alpha}, seed={seed} ===")
    print(f"  Proxy:   {len(proxy_indices):>6,} ({proxy_per_class}/class)")
    print(f"  Private: {len(private_indices):>6,}")
    for k in range(num_clients):
        cl = labels[client_indices[k]]
        n_cls = len(np.unique(cl))
        top3 = np.argsort(np.bincount(cl, minlength=num_classes))[-3:][::-1]
        top3_counts = [int(np.sum(cl == c)) for c in top3]
        print(f"  Client {k}: {len(client_indices[k]):>6,}장  "
              f"활성 클래스 {n_cls:>3}개  "
              f"top3: {list(zip(top3.tolist(), top3_counts))}")

    # 5) 저장
    save_dict = {
        "proxy_indices": proxy_indices,
        "private_indices": private_indices,
        "alpha": np.array([alpha]),
        "seed": np.array([seed]),
        "num_clients": np.array([num_clients]),
        "num_classes": np.array([num_classes]),
        "proxy_per_class": np.array([proxy_per_class]),
    }
    for k in range(num_clients):
        save_dict[f"client_{k}"] = client_indices[k]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **save_dict)
    print(f"  saved → {out_path}")


# =============================================================================
# 메인
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=None,
                        help="단일 α (없으면 --all과 함께 전체)")
    parser.add_argument("--seed", type=int, default=None,
                        help="단일 seed")
    parser.add_argument("--all", action="store_true",
                        help="모든 α × seed 조합 생성")
    parser.add_argument("--force", action="store_true",
                        help="기존 파티션 덮어쓰기")
    args = parser.parse_args()

    ensure_dirs()

    # 데이터셋 로드 — labels만 필요하므로 transform 불필요
    # 단, parquet 로딩 자체는 느리므로 한 번만 하고 재사용
    print(f"Loading ImageNet-100 train parquet from {DATA_ROOT} ...")
    table = load_parquet_table("train")
    dataset = ParquetImageDataset(transform=None, shared_table=table)
    print(f"  Total train samples: {len(dataset):,}")
    print(f"  Unique labels: {len(set(dataset.labels))}")

    if args.all:
        combos = [(a, s) for a in ALPHAS for s in SEEDS]
    elif args.alpha is not None and args.seed is not None:
        combos = [(args.alpha, args.seed)]
    else:
        parser.error("--all 또는 --alpha와 --seed를 모두 지정")

    for alpha, seed in combos:
        build_partition(alpha, seed, dataset, force=args.force)

    print("\n=== Partition 생성 완료 ===")


if __name__ == "__main__":
    main()
