#!/usr/bin/env bash
# =============================================================================
# run_all.sh — 단일 EC2에서 전체 파이프라인 실행 (기준 스크립트)
# =============================================================================
# 이 스크립트는 모든 단계를 순차 실행합니다. EC2 한 대로 처리할 경우 사용.
# 병렬 실행은 run_bounds.sh / run_teachers.sh / run_kd.sh 를 별도 EC2에 배포.
#
# 사용:
#   bash run_all.sh
#
# 환경변수 (선택):
#   EACS_RESULTS_ROOT — 결과 디렉토리 (기본 ~/eacs_results)
#   EACS_DATA_ROOT    — ImageNet-100 경로 (기본 ~/data/imagenet100)
#
# EC2 종료 패턴:
#   nohup bash -c 'bash run_all.sh > run_all.log 2>&1 ; sudo shutdown -h now' &

set -e  # 에러 시 중단

ALPHAS=(0.1 0.5 1.0 10.0 100.0)
SEEDS=(42 123)
WEIGHTINGS=(uniform top_1 top_3)

echo "====================================="
echo "Step 1: Prepare partitions"
echo "====================================="
python prepare_partition.py --all

echo "====================================="
echo "Step 2: Train bounds (lower + upper, seed별)"
echo "====================================="
for seed in "${SEEDS[@]}"; do
    python train_bounds.py --seed $seed --mode both
done

echo "====================================="
echo "Step 3: Train teachers (α × seed)"
echo "====================================="
for alpha in "${ALPHAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        python train_teachers.py --alpha $alpha --seed $seed
    done
done

echo "====================================="
echo "Step 4: Run KD (α × seed × weighting)"
echo "====================================="
for alpha in "${ALPHAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for w in "${WEIGHTINGS[@]}"; do
            python run_kd.py --alpha $alpha --seed $seed --weighting $w
        done
    done
done

echo "====================================="
echo "Step 5: Aggregate + Visualize"
echo "====================================="
python aggregate_sweep.py

echo "====================================="
echo "완료! figures는 \$EACS_RESULTS_ROOT/figures 에 있습니다."
echo "====================================="
