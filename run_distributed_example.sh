#!/usr/bin/env bash
# =============================================================================
# run_distributed_example.sh — EC2 병렬 분산 실행 예시
# =============================================================================
# 이 파일은 "예시"입니다. 실제로는 각 EC2에서 필요한 부분만 복사해서 쓰세요.
#
# 전제:
#   - EFS를 통해 모든 EC2가 같은 RESULTS_ROOT를 공유함
#   - prepare_partition.py는 어느 EC2에서든 한 번만 먼저 실행
#     (매우 빠르므로 Step 1 끝나기를 기다리고 나머지 시작)
#
# 권장 분산 전략 (5대 EC2 기준, 총 ~4일 내외 완료):
#
#   EC2 #1 (Bounds 전담, seed=42+123):
#     이 EC2는 Lower + Upper만 돌림. 다른 EC2와 겹치지 않음.
#     약 22h 소요.
#
#   EC2 #2, #3, #4, #5 (Teacher + KD 병렬):
#     α를 나눠서 Teacher와 KD를 순차로 처리.
#
# =============================================================================
# 공통 선행 작업: 모든 EC2에서 한 번만 (누구든)
# =============================================================================
# python prepare_partition.py --all


# =============================================================================
# EC2 #1 — Bounds 전담
# =============================================================================
# nohup bash -c '
#   python train_bounds.py --seed 42 --mode both ;
#   python train_bounds.py --seed 123 --mode both ;
#   sudo shutdown -h now
# ' > ec2_1_bounds.log 2>&1 &


# =============================================================================
# EC2 #2 — α={0.1, 0.5}, seed=42,123 (Teacher + KD)
# =============================================================================
# nohup bash -c '
#   for alpha in 0.1 0.5; do
#     for seed in 42 123; do
#       python train_teachers.py --alpha $alpha --seed $seed
#       for w in uniform top_1 top_3; do
#         python run_kd.py --alpha $alpha --seed $seed --weighting $w
#       done
#     done
#   done ;
#   sudo shutdown -h now
# ' > ec2_2.log 2>&1 &


# =============================================================================
# EC2 #3 — α={1.0}, seed=42,123
# =============================================================================
# nohup bash -c '
#   for alpha in 1.0; do
#     for seed in 42 123; do
#       python train_teachers.py --alpha $alpha --seed $seed
#       for w in uniform top_1 top_3; do
#         python run_kd.py --alpha $alpha --seed $seed --weighting $w
#       done
#     done
#   done ;
#   sudo shutdown -h now
# ' > ec2_3.log 2>&1 &


# =============================================================================
# EC2 #4 — α={10.0}, seed=42,123
# =============================================================================
# nohup bash -c '
#   for alpha in 10.0; do
#     for seed in 42 123; do
#       python train_teachers.py --alpha $alpha --seed $seed
#       for w in uniform top_1 top_3; do
#         python run_kd.py --alpha $alpha --seed $seed --weighting $w
#       done
#     done
#   done ;
#   sudo shutdown -h now
# ' > ec2_4.log 2>&1 &


# =============================================================================
# EC2 #5 — α={100.0}, seed=42,123
# =============================================================================
# nohup bash -c '
#   for alpha in 100.0; do
#     for seed in 42 123; do
#       python train_teachers.py --alpha $alpha --seed $seed
#       for w in uniform top_1 top_3; do
#         python run_kd.py --alpha $alpha --seed $seed --weighting $w
#       done
#     done
#   done ;
#   sudo shutdown -h now
# ' > ec2_5.log 2>&1 &


# =============================================================================
# 집계 (모든 EC2 완료 후, 한 대에서)
# =============================================================================
# python aggregate_sweep.py

# =============================================================================
# 추가 팁
# =============================================================================
# 1. TMPDIR 설정 (ImageNet 로딩 성능 향상):
#    export TMPDIR=/tmp  # EFS가 아닌 로컬 NVMe
#
# 2. PyTorch 환경 활성화 (AWS DL AMI 기준):
#    source /opt/pytorch/bin/activate
#
# 3. HF_HOME / TORCH_HOME을 EFS에 두면 여러 EC2가 모델 weights 공유 가능:
#    export HF_HOME=$EACS_RESULTS_ROOT/../hf_cache
#
# 4. 실행 중 진행상황 확인:
#    tail -f ec2_X.log
#    ls $EACS_RESULTS_ROOT/teachers/  # 완료된 (α,seed) 확인
#    ls $EACS_RESULTS_ROOT/kd_runs/   # 완료된 KD 확인
#
# 5. 중단된 EC2 재개:
#    각 스크립트는 이미 존재하는 결과를 SKIP하므로 재실행 안전함.
#    --force 플래그 쓰면 강제 재학습.
