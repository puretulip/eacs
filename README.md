# Non-IID KD 구조적 한계 검증 — 실험 코드

본 코드는 `non_iid_kd_limit_report.md` 보고서의 실험 설계를 구현합니다.

## 논지와 구성

증명하려는 것을 두 Phase로 분리했습니다:

### Phase 1 (메인) — 주장 A: "Non-IID Teacher의 logit은 dark knowledge가 없으면 noise다"

- **동일 모델 구성**: Teacher = Student = Lower = Upper = **ResNet-18 (ImageNet-1K pretrained)**
- 모델 크기/아키텍처 변수 완전 제거 → 데이터 접근 방식만 독립 변수
- **Layer 2 지표 (logit entropy, top-1 confidence)가 핵심 증거**
- α 5점: {0.1, 0.5, 1.0, 10.0, 100.0}

### Phase 2 (보조) — 주장 B: "Small→Large 현실 시나리오에서 KD 실패"

- **Small→Large 구성**: Teacher = MobileNetV2 pretrained, Student = Lower = Upper = **ResNet-50 pretrained**
- 엣지(작은 모델) → 서버(큰 모델) 지식 전달 시나리오
- **Gap Recovery가 음수로 떨어지는지**가 핵심 증거
- α 3점: {0.1, 1.0, 100.0} (Phase 2는 축소)

## 공통 설정

- **데이터셋**: ImageNet-100 (HuggingFace parquet)
- **K (Teacher 수)**: 5
- **Proxy per class**: 100
- **Weighting**: {uniform, top_1, top_3}
- **Seed**: {42, 123}
- **KD 설정**: kd_alpha=0.3 (CE 30% + KD 70%), Temperature=4.0
- **Optimizer**: SGD, LR=0.01, WD=1e-4, momentum=0.9 (pretrained fine-tuning 표준)

## 파일 구조

```
common.py                      공통 유틸 (3개 모델 빌더 + build_model_for_role 헬퍼)
prepare_partition.py           파티션 생성 (Phase 무관, 한 번만 실행)
train_bounds.py                Lower/Upper Bound 학습 (--phase 1 or 2)
train_teachers.py              Teacher 학습 + Logit 수집 (--phase 1 or 2)
run_kd.py                      Multi-Teacher KD 학습 (--phase 1 or 2)
aggregate_sweep.py             결과 집계 + Figure 생성 (Phase별)
README.md                      본 가이드
```

## 저장 디렉토리 구조

```
$EACS_RESULTS_ROOT/
├── partitions/                         (Phase 무관, 공유)
│   └── alpha{α}_seed{s}.npz
├── logs/
├── phase1/                             (주장 A, 동일 ResNet-18)
│   ├── bounds/seed{s}/
│   ├── teachers/alpha{α}_seed{s}/
│   ├── kd_runs/alpha{α}_seed{s}_{w}/
│   ├── logs/
│   └── figures/                        (phase1 전용 Figure)
└── phase2/                             (주장 B, Small→Large)
    ├── bounds/seed{s}/
    ├── teachers/alpha{α}_seed{s}/
    ├── kd_runs/alpha{α}_seed{s}_{w}/
    ├── logs/
    └── figures/                        (phase2 전용 Figure)
```

## 환경 준비

```bash
# 필요 패키지
pip install torch torchvision numpy pandas matplotlib seaborn pillow pyarrow

# 환경변수
export EACS_RESULTS_ROOT=/mnt/efs/eacs_results    # 결과 저장 (EFS)
export EACS_DATA_ROOT=/mnt/efs/data/imagenet100   # ImageNet-100 parquet
export TMPDIR=/tmp                                # 로컬 NVMe

# PyTorch 환경 활성화 (AWS DL AMI)
source /opt/pytorch/bin/activate
```

데이터 디렉토리 구조:
```
$EACS_DATA_ROOT/
├── train.parquet           # HuggingFace ImageNet-100 train
├── validation.parquet      # HuggingFace ImageNet-100 validation
└── class_mapping.txt       # (선택) 클래스 인덱스 → 이름
```

## 실행 순서

### 1. 파티션 생성 (Phase 공용, 한 번만)

```bash
python prepare_partition.py --all
```

출력 마지막에 다음이 나와야 함:
```
✓ 모든 파티션이 동일한 proxy 사용 (checksum=...)
```

### 2. Pilot 실행 (권장) — 약 5-8시간

핵심 패턴이 관측되는지 먼저 확인:

```bash
# Phase 1 Pilot
python train_bounds.py --phase 1 --seed 42 --mode both
python train_teachers.py --phase 1 --alpha 0.1 --seed 42
python run_kd.py --phase 1 --alpha 0.1 --seed 42 --weighting uniform
python run_kd.py --phase 1 --alpha 0.1 --seed 42 --weighting top_1
python run_kd.py --phase 1 --alpha 0.1 --seed 42 --weighting top_3
```

**Phase 1 Pilot 성공 조건**:
- Upper Bound ~85%
- Lower Bound ~82-85%
- α=0.1의 Teacher logit entropy가 IID 대비 확연히 낮음 (Layer 2)

### 3. 본 실험 — Phase 1 (메인)

**Bounds 학습** (seed당 1회씩):
```bash
python train_bounds.py --phase 1 --seed 42 --mode both
python train_bounds.py --phase 1 --seed 123 --mode both
```

**Teacher 학습** (10조합):
```bash
for alpha in 0.1 0.5 1.0 10.0 100.0; do
  for seed in 42 123; do
    python train_teachers.py --phase 1 --alpha $alpha --seed $seed
  done
done
```

**KD 학습** (30조합):
```bash
for alpha in 0.1 0.5 1.0 10.0 100.0; do
  for seed in 42 123; do
    for w in uniform top_1 top_3; do
      python run_kd.py --phase 1 --alpha $alpha --seed $seed --weighting $w
    done
  done
done
```

### 4. 본 실험 — Phase 2 (보조)

**Bounds** (ResNet-50라 오래 걸림):
```bash
python train_bounds.py --phase 2 --seed 42 --mode both
python train_bounds.py --phase 2 --seed 123 --mode both
```

**Teacher 학습** (α 3점 × seed 2개 = 6조합):
```bash
for alpha in 0.1 1.0 100.0; do
  for seed in 42 123; do
    python train_teachers.py --phase 2 --alpha $alpha --seed $seed
  done
done
```

**KD 학습** (18조합):
```bash
for alpha in 0.1 1.0 100.0; do
  for seed in 42 123; do
    for w in uniform top_1 top_3; do
      python run_kd.py --phase 2 --alpha $alpha --seed $seed --weighting $w
    done
  done
done
```

### 5. 집계 및 시각화

```bash
# Phase 1 + 2 모두 집계 (기본)
python aggregate_sweep.py

# 특정 Phase만
python aggregate_sweep.py --phase 1
python aggregate_sweep.py --phase 2
```

결과는 `$EACS_RESULTS_ROOT/phase{1,2}/figures/`에 저장됩니다.

## EC2 분산 실행 예시

모든 스크립트는 **이미 존재하는 결과를 SKIP**하므로 병렬 실행 안전. 예시 분배:

| EC2 | 담당 |
|---|---|
| A | Phase 1 Bounds (seed 42, 123) |
| B | Phase 1, α={0.1, 0.5} 전 조합 |
| C | Phase 1, α={1.0, 10.0} 전 조합 |
| D | Phase 1, α=100.0 + Phase 2 Bounds |
| E | Phase 2, α={0.1, 1.0, 100.0} 전 조합 |

**EC2 자동 셧다운 패턴**:
```bash
nohup bash -c '
  python train_bounds.py --phase 1 --seed 42 --mode both
  python train_bounds.py --phase 1 --seed 123 --mode both
  sudo shutdown -h now
' > ec2_A.log 2>&1 &
```

## 시간 예산 (g5.2xlarge 기준)

### Phase 1 (ResNet-18 pretrained)
- Bounds (seed 2개): ~8h
- Teacher 학습 (50회): ~50h (Teacher당 ~1h)
- KD (30회): ~30h (KD당 ~1h)
- **Phase 1 합계: ~88h (~3.7일)**

### Phase 2 (ResNet-50 pretrained, 축소 α)
- Bounds (seed 2개): ~16h
- Teacher 학습 (30회, MobileNetV2): ~25h
- KD (18회): ~30h (ResNet-50 Student)
- **Phase 2 합계: ~71h (~3일)**

### 전체 합계
- 단일 EC2 순차 실행: ~160h (~6.5일)
- 5 EC2 분산: **~3~4일**

## 핵심 Figure 해석 가이드

### Phase 1 (주장 A)

- **Figure 3a-3c (메인)**: α↓일수록 logit entropy↓ / top-1 CDF 오른쪽 쏠림 → "dark knowledge 부재 = noise" 직접 증거
- **Figure 4b**: Gap이 작아서 Student acc 차이는 작을 수 있음. 그래도 α 방향성은 보여야 함
- **Figure 4c**: Gap Recovery가 α↓일수록 감소

### Phase 2 (주장 B)

- **Figure 4b**: KD 결과가 Lower Bound 아래로 떨어지면 핵심 증거
- **Figure 4c**: Gap Recovery 음수 진입이 구조적 한계의 최종 증거
- **Figure 3a-3c**: Phase 1과 함께 비교하여 "같은 원리가 시나리오를 가리지 않는다"는 보조 증거

## 주요 하이퍼파라미터

| 파라미터 | 값 | 위치 |
|---|---|---|
| Teacher epochs | 30 | `train_teachers.py` |
| Lower Bound epochs | 40 | `train_bounds.py` |
| Upper Bound epochs | 25 | `train_bounds.py` |
| KD epochs | 40 | `run_kd.py` |
| KD alpha | 0.3 (CE 30% + KD 70%) | `run_kd.py` |
| KD temperature | 4.0 | `run_kd.py` |
| Learning rate | 0.01 (pretrained fine-tune) | 전 스크립트 |
| Batch size Phase 1 | 128 | ResNet-18 |
| Batch size Phase 2 | 64 | ResNet-50 |
| Proxy per class | 100 | `prepare_partition.py` |

## 에폭 시간 로그

모든 학습은 `logs/` 하위에 JSON-lines 형식으로 에폭 시간 기록:

```
{"tag": "p1_teacher_k0_a0.1_s42", "epoch": 0, "elapsed_sec": 120.3}
{"tag": "p1_teacher_k0_a0.1_s42", "epoch": 1, "elapsed_sec": 118.7}
...
{"SUMMARY": {"tag": "p1_teacher_k0_a0.1_s42", "n_epochs": 30, "total_sec": 3543, ...}}
```

## 중단/재개

- 모든 스크립트는 **완료 판정 파일**(`metrics.json` 또는 `metadata.json`) 기준으로 SKIP 결정
- `.pt` ckpt 파일만 있고 완료 파일이 없으면 **미완료 상태로 간주하여 재학습**
- `--force` 플래그로 강제 재학습 가능

## 문제 발생 시

1. **OOM**: Phase 2의 batch_size를 더 줄이기 (`run_kd.py`, `train_bounds.py`의 `BATCH_SIZE_BY_PHASE`)
2. **느린 데이터 로딩**: `TMPDIR=/tmp` 확인 (로컬 NVMe)
3. **EFS 쓰기 경합**: 같은 (phase, α, seed) 조합을 두 EC2에서 동시 실행하지 않기
4. **Proxy 불일치 오류**: `prepare_partition.py --all --force`로 전체 재생성
