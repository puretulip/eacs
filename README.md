# Non-IID KD 구조적 한계 검증 — 실험 코드

본 코드는 `non_iid_kd_limit_report.md` 보고서의 실험 설계를 구현한 것입니다.

## 설계 요약

- **데이터셋**: ImageNet-100
- **모델**: ResNet-18 (Teacher=Student, scratch)
- **K (Teacher 수)**: 5
- **α (Dirichlet)**: {0.1, 0.5, 1.0, 10.0, 100.0}
- **Weighting**: {uniform, top_1, top_3}
- **Seed**: {42, 123}

총 실험: Lower/Upper × 2 seed + Teacher 학습 5α×2seed×5명 + KD 5α×2seed×3w = **30 KD runs**

## 파일 구조

```
common.py                      공통 유틸 (데이터셋, 모델, 시간 로거, 지표 계산)
prepare_partition.py           파티션 생성 (빠름, 한 번만 실행)
train_bounds.py                Lower/Upper Bound 학습 (α 무관)
train_teachers.py              Teacher 학습 + Logit 사전 수집 + Layer 2 지표
run_kd.py                      Multi-Teacher KD 학습
aggregate_sweep.py             결과 집계 + 모든 Figure 생성
run_all.sh                     단일 EC2용 전체 실행
run_distributed_example.sh     EC2 여러 대 분산 실행 템플릿
```

## 실행 순서

### 0. 환경 준비

```bash
# 필요 패키지
pip install torch torchvision numpy pandas matplotlib seaborn pillow pyarrow

# 환경변수 설정
export EACS_RESULTS_ROOT=/mnt/efs/eacs_results    # 결과 저장 (EFS 공유 경로)
export EACS_DATA_ROOT=/mnt/efs/data/imagenet100   # ImageNet-100 parquet 경로
export TMPDIR=/tmp                                # 로컬 NVMe (PIL 임시파일)
```

**데이터 디렉토리 구조**:

```
$EACS_DATA_ROOT/
├── train.parquet           # HuggingFace ImageNet-100 train
├── validation.parquet      # HuggingFace ImageNet-100 validation
└── class_mapping.txt       # (선택) 클래스 인덱스 → 이름 매핑
```

Parquet 포맷 (HuggingFace `datasets` 표준):
- `image` 컬럼: JPEG/PNG bytes (또는 `{'bytes': ..., 'path': ...}` dict)
- `label` 컬럼: int (0..99)

### 1. 파티션 생성 (한 번만, 매우 빠름)

```bash
python prepare_partition.py --all
```

### 2. Bounds 학습 (어느 EC2든 seed별로 1번씩)

```bash
python train_bounds.py --seed 42 --mode both
python train_bounds.py --seed 123 --mode both
```

### 3. Teacher 학습 (α × seed마다)

```bash
python train_teachers.py --alpha 0.1 --seed 42
# ... 10개 조합 반복
```

### 4. KD 학습 (α × seed × weighting)

```bash
python run_kd.py --alpha 0.1 --seed 42 --weighting uniform
# ... 30개 조합 반복
```

### 5. 집계 및 시각화

```bash
python aggregate_sweep.py
# 결과: $EACS_RESULTS_ROOT/figures/
```

## EC2 분산 실행

`run_distributed_example.sh` 참고. 권장:

- **EC2 A**: Bounds 전담 (seed 42, 123)
- **EC2 B~F**: α별 분배 — 각 EC2가 자기 α의 Teacher+KD 처리

모든 스크립트는 **이미 존재하는 결과를 SKIP**하므로 병렬 실행 간 충돌 없음.
`--force` 플래그로 강제 재학습 가능.

## 결과물

### 저장 구조

```
$EACS_RESULTS_ROOT/
├── partitions/alpha{α}_seed{s}.npz
├── bounds/seed{s}/{lower,upper}.pt + {lower,upper}_metrics.json
├── teachers/alpha{α}_seed{s}/
│   ├── teachers.pt              # K=5 Teacher weights
│   ├── teacher_logits.pt        # proxy 상 logit (KD 재사용)
│   ├── expertise.npz            # F1/P/R 행렬
│   └── metadata.json            # Teacher val acc, coverage, logit quality
├── kd_runs/alpha{α}_seed{s}_{w}/
│   ├── student_best.pt
│   └── metrics.json             # accuracy, gap_recovery, per-class, history
├── logs/*.log                    # 에폭별 소요 시간 로그
└── figures/                      # aggregate_sweep.py가 생성
    ├── fig1a_partition_heatmap.png
    ├── fig1b_active_classes.png
    ├── fig1c_total_samples.png
    ├── fig2a_expertise_heatmap.png
    ├── fig2b_teacher_boxplot.png
    ├── fig3a_entropy_violin.png
    ├── fig3b_top1_cdf.png
    ├── fig3c_expert_vs_nonexpert_entropy.png
    ├── fig4a_main_inverse_correlation.png  ← 메인 증거
    ├── fig4b_weighting_comparison.png
    ├── fig4c_gap_recovery.png
    ├── fig4d_per_class_alpha01.png
    ├── summary_raw.csv
    └── summary_aggregated.csv
```

### 에폭 시간 로그

`logs/` 디렉토리에 JSON-lines 형식으로 에폭별 시간이 기록됩니다:

```
{"tag": "teacher_k0_a0.1_s42", "epoch": 0, "elapsed_sec": 87.3}
{"tag": "teacher_k0_a0.1_s42", "epoch": 1, "elapsed_sec": 85.9}
...
{"SUMMARY": {"tag": "teacher_k0_a0.1_s42", "n_epochs": 60, "total_sec": 5178, ...}}
```

## 주요 하이퍼파라미터

공통 모듈(`common.py`) 및 각 스크립트 상단에서 조정 가능:

| 파라미터 | 값 | 위치 |
|---|---|---|
| Teacher epochs | 60 | `train_teachers.py` |
| Lower Bound epochs | 100 | `train_bounds.py` |
| Upper Bound epochs | 60 | `train_bounds.py` |
| KD epochs | 80 | `run_kd.py` |
| KD α (CE 비중) | 0.5 | `run_kd.py` |
| KD temperature | 4.0 | `run_kd.py` |
| Batch size | 128 | 전 스크립트 |
| Learning rate | 0.1 (SGD) | 전 스크립트 |
| Proxy per class | 100 | `prepare_partition.py` |

## 핵심 Figure 해석 가이드

### Figure 4a (메인) — 역상관 관측
- 주황 곡선(Teacher F1, 좌측 y축)과 파랑 곡선(Student Acc, 우측 y축)이 **역방향**으로 움직이면 가설 검증 성공
- α↓ 방향으로 Teacher F1 ↑, Student Acc ↓가 나와야 함

### Figure 4b — Top-1의 역설
- α=0.1에서 Top-1(빨강)이 Uniform(파랑) 아래로 떨어지면 "F1 기반 선별이 역효과" 증거
- α=100에서 세 곡선이 거의 겹치면 "IID에서는 선별이 무의미" 증거

### Figure 3a, 3b — Dark Knowledge 부재
- α↓일수록 entropy violin이 아래로 납작해지고 확장됨
- α↓일수록 top-1 CDF가 오른쪽(1.0)에 쏠림
- → Teacher logit이 hard label화됨을 직접 보임

## 문제 발생 시

1. **Out of memory**: `--no-amp` 끄지 말고 batch_size 조정 (common 내 상수)
2. **느린 데이터 로딩**: `TMPDIR=/tmp` 확인 (로컬 NVMe에 PIL 임시파일)
3. **EFS 쓰기 경합**: 같은 (α, seed)를 두 EC2가 동시에 학습하지 않도록 주의
4. **Teacher 학습이 수렴 안 함**: α=0.1 극단 조건에서는 일부 클래스 데이터가 매우 적음. 정상
