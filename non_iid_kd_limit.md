# Non-IID Federated KD의 구조적 한계 검증 — 실험 설계 및 예상 결과 보고

> **작성일**: 2026-04-24
> **문서 성격**: 지도교수 보고용 내부 기술 보고서
> **목적**: 그간 축적된 비판적 분석(critical review v2)과 실험 결과(v4 regression, single-teacher KD ~0.686)를 바탕으로, "Cross-silo + Non-IID 환경에서 logit 기반 KD가 구조적으로 작동하지 않는다"는 가설을 실험적으로 검증하기 위한 설계안
> **관련 문서**: `eacs_kd_critical_review_v2.md`, `experiment_design_expertise_aware_distillation.md`, `eacs_kd_multi_teacher.py`

---

## 0. 요약

본 보고서는 EaCS-KD 연구의 방향 전환을 위한 근거 확보 실험을 제안한다. 기존 FL-KD 연구들이 암묵적으로 전제하는 두 가지 가정 — **(A) Teacher의 분류 성능이 높으면 KD 소스로도 유용하다**, **(B) Non-IID로 학습된 Teacher의 logit에 dark knowledge가 담겨 있다** — 이 두 가정이 극단적 Non-IID 환경에서 성립하지 않음을 단일 실험 프레임에서 보이고자 한다.

핵심은 **α (Dirichlet 집중도)를 축으로 한 스윕 실험**이다. α를 IID(α=100)에서 극단적 Non-IID(α=0.1)까지 변화시키면서 다음 세 가지를 동시에 측정한다:

1. **개별 Teacher의 per-class F1** — Teacher의 분류 성능
2. **Teacher logit의 품질 지표 (entropy, top-1 confidence)** — dark knowledge 유무
3. **Student의 global accuracy (Uniform / Top-1 / Top-3 가중치별)** — KD 결과물

예상되는 결과는 **개별 Teacher F1과 Student global accuracy 간의 역상관** 이다. 즉, 개별 Teacher의 성능이 가장 높을 때(α=0.1) Student의 성능은 가장 낮게 나오며, 개별 Teacher의 성능이 평범할 때(α=100, IID) Student의 성능은 가장 높게 나올 것으로 예상된다. 이는 직관적으로 "잘 학습된 Teacher가 KD에 유리해야 한다"는 전제를 뒤집는 결과이며, Non-IID KD의 기반 가정이 잘못되었다는 강력한 근거가 된다.

---

## 1. 연구 현 위치와 본 실험의 목적

### 1.1 그동안 누적된 관찰

| 실험 | 조건 | 결과 | 시사점 |
|---|---|---|---|
| v3_2 | Pretrained backbone, α=0.1 | KD 후 성능 하락 | Student가 강하면 Non-IID Teacher logit은 오염원 |
| v4 | Scratch, Small→Large, α=0.1 | Naive KD < Lower Bound | 필터링 없는 KD는 음의 gap recovery |
| v5 (현재) | Scratch, 동일 크기, α=0.1 | Uniform > Top-K F1 | 선별이 오히려 역효과 (Top-K의 hard label화) |
| Single-Teacher KD | IID full data Teacher → Student | ~0.686 (실용적 상한) | Multi-teacher Non-IID 구성들이 모두 이보다 낮음 |

이 네 관찰은 각각 다른 맥락에서 얻어졌지만, 하나의 가설로 수렴한다: **극단적 Non-IID Teacher의 logit에는 KD가 전달해야 할 dark knowledge가 구조적으로 존재하지 않는다.**

`critical_review_v2` 문서는 이 가설에 대한 이론적 분석을 제공하지만, 이 가설을 **단일 통제된 실험 프레임** 내에서 체계적으로 실증한 결과는 아직 없다. 본 실험의 목적이 바로 그것이다.

### 1.2 본 실험이 증명하려는 것

**주장 1 (가정 A의 반박)**
개별 Teacher의 per-class F1이 높은 것이 KD 성공의 충분조건도 필요조건도 아니다. 오히려 Non-IID가 심해질수록 개별 F1은 올라가지만 KD 결과는 나빠진다.

**주장 2 (가정 B의 반박)**
Non-IID Teacher의 logit은 dark knowledge가 부재한 분포다. α가 낮아질수록 전문 클래스 logit의 entropy가 급감하고 top-1 confidence가 1에 수렴한다(hard-label화). 비전문 클래스 logit은 학습된 적 없는 noise다.

**주장 3 (가중치 전략의 한계)**
Uniform, Top-1, Top-3 어떤 가중치 방법도 주장 1, 2를 회피하지 못한다. Top-1은 hard-label화를 가속하고, Uniform은 noise를 평균하지만 fundamental signal 품질 자체를 개선하지는 못한다.

### 1.3 본 실험이 증명하지 않는 것 (scope)

- 본 실험은 **Non-IID + logit KD 조합**의 한계를 보이는 것이지, KD 자체 또는 FL 자체의 한계를 보이는 것이 아니다.
- Feature-level KD, prototype 기반 방법, parameter averaging(FedAvg) 등은 본 실험의 직접 대상이 아니다. 이들에 대한 비교는 후속 보고에서 다룬다.
- 본 실험은 단일 데이터셋(ImageNet-100) + 단일 모델(ResNet-18) 조건에서의 검증이며, 도메인 일반화는 scope 밖이다.

---

## 2. 해체 대상이 되는 두 가정

### 2.1 가정 A — "Teacher 성능 = KD 소스 품질"

전통적 KD(Hinton 2015)는 **하나의 잘 학습된 Teacher**를 가정한다. 이 Teacher는 IID 데이터로 학습되었으므로 모든 클래스에 대해 균형 잡힌 예측 분포를 가진다. 이 설정에서 Teacher accuracy ↑ = soft label 품질 ↑ = Student 학습 효과 ↑ 의 인과 사슬이 성립한다.

FL-KD 연구들은 이 인과를 **Non-IID 환경으로 그대로 확장**한다. 즉, 각 Teacher가 자기 로컬 데이터로 잘 학습되면(per-class F1 ↑) 그 Teacher의 logit도 유용한 KD 소스가 될 것이라고 본다. EaCS-KD 자체도 이 가정 위에서 "F1 기반 가중치"를 설계했다.

하지만 이 확장은 부당하다. Non-IID Teacher의 높은 F1은 **일부 클래스에 국한된 높은 F1**이며, 이는 IID Teacher의 전역 F1과 구조적으로 다르다. 구체적으로:

| IID Teacher의 높은 F1 | Non-IID Teacher의 높은 F1 (전문 클래스 한정) |
|---|---|
| 모든 클래스 간 경쟁 속에서 구별을 학습 | 해당 클래스가 사실상 단독으로 존재 |
| 클래스 간 유사성 정보가 logit에 반영됨 | 해당 클래스로 거의 모든 confidence가 집중 |
| Soft label의 entropy가 살아 있음 | Soft label이 one-hot에 수렴 |
| Student가 다중 클래스 관계를 배울 수 있음 | Student에게 전달되는 건 사실상 hard label |

따라서 "F1이 높은 Teacher = 좋은 KD 소스"라는 등식은 Non-IID 환경에서 깨진다. 이것을 실험으로 보이는 것이 본 연구의 첫 번째 목표다.

### 2.2 가정 B — "Non-IID Teacher logit에 dark knowledge 존재"

Dark knowledge는 Teacher의 soft label에 담긴 **클래스 간 관계 정보**다. "고양이 0.7 / 호랑이 0.15 / 개 0.1"이라는 분포에서 "고양이와 호랑이는 시각적으로 유사하다"는 정보가 Student에게 전달된다.

이 정보가 존재하려면 Teacher가 학습 시 **모든 관련 클래스를 동시에 구별해야 할 상황에 노출**되었어야 한다. 해당 클래스를 거의 보지 못한 Teacher에게 그 클래스에 대한 logit을 요구하는 것은, 본 적 없는 문제에 대한 확률 분포를 요구하는 것과 같다 — 그 분포는 정보가 아니라 noise다.

그런데 Non-IID 환경에서는 정의상 Teacher마다 본 클래스의 집합이 제한된다. 전문 클래스의 logit은 **극단적으로 sharp해지고**(one-hot에 수렴), 비전문 클래스의 logit은 **임의의 noise**가 된다. 둘 다 dark knowledge가 아니다.

이것을 실험으로 보이려면 단순히 Student accuracy를 측정하는 것으로는 부족하다. **logit의 품질 자체**를 직접 측정해야 한다. 구체적으로 각 α 조건에서:

- Teacher logit의 평균 entropy (low → hard-label화)
- Teacher logit의 top-1 confidence 분포 (1에 수렴 → dark knowledge 소실)
- 전문 클래스 logit과 비전문 클래스 logit 각각의 top-2 gap

이 세 지표가 α 감소와 함께 어떻게 변하는지를 보이면, dark knowledge 부재를 직접적으로 증명할 수 있다.

---

## 3. 실험 설계

### 3.1 데이터 파티션 구조

**기존 `eacs_kd_multi_teacher.py`의 파티션 방식을 유지한다.**

```
전체 train set
    │
    ├─ (1) Proxy 분리: 클래스당 P장 균등 추출 → 서버 보유 (Teacher는 접근 불가)
    │         P는 --proxy_per_class 인자로 제어 (기본 100)
    │
    └─ (2) 나머지 Private pool → Dirichlet(α)로 K개 Teacher에게 분배
              각 Teacher는 자기 몫의 private data만으로 학습
```

이 구조는 FedMD/FedDF/FedLAW 표준과 일치하며, 다음을 보장한다:
- Proxy data는 Teacher 학습에 노출되지 않으므로, proxy 상의 Teacher 성능은 **일반화된 성능**으로 해석 가능
- Expertise Scoring (F1 측정)이 unbiased
- Student 학습 시 Teacher logit은 "본 적 없는 샘플에 대한 예측"이므로 KD의 전제 성립

### 3.2 실험 변수 매트릭스

| 축 | 값 | 목적 |
|---|---|---|
| **Dirichlet α** | {0.1, 0.5, 1.0, 10.0, 100.0} | Non-IID 강도 스윕 (핵심 축) |
| **Weighting 방법** | {Uniform, Top-1 F1, Top-3 F1} | 가중치 전략이 α 변화에 어떻게 반응하는가 |
| **Proxy size (per class)** | {100} 고정 | 본 실험의 주 변수가 아님 (ablation 시 추가 가능) |
| **Teacher 수 (K)** | 5 고정 | Non-IID 효과가 충분히 드러나는 최소 수 |
| **Seed** | {42, 123, 2024} | 3-seed 평균 ± 표준편차 보고 |

**총 실험 횟수**: 5(α) × 3(weighting) × 3(seed) = **45회 KD 학습**
별도로 α당 Teacher 학습(5개) × 5(α) × 3(seed) = **75회 Teacher 학습**
단, Teacher와 Upper/Lower Bound는 α 변화와 무관하거나 α에 따라서만 달라지므로 캐싱으로 상당 부분 절감 가능.

**Lower/Upper Bound**는 α에 무관하므로 seed당 1회씩 총 3회만 측정하면 된다.

### 3.3 Top-K F1 방식의 명시

본 실험에서 "Top-1 F1"과 "Top-3 F1"은 기존 `eacs_kd_multi_teacher.py`의 vector-level Top-K 가중 합산을 의미한다:

```
각 클래스 c에 대해:
  1. K명의 Teacher 중 F1(k, c)가 상위인 n명 선택 (n=1 또는 3)
  2. 선택된 n명의 F1을 정규화하여 가중치로 사용
  3. 해당 가중치로 Teacher logit vector 전체를 가중 합산
```

**Piecewise 방식은 본 실험 scope에 포함하지 않는다.** 이유: 본 실험의 목적은 "가중치 전략의 우열 비교"가 아니라 "α 변화가 KD 전반에 미치는 구조적 영향"이다. Piecewise는 별개의 ablation 질문이며, 혼재시키면 해석 축이 2개가 되어 보고서 메시지가 흐려진다.

### 3.4 측정 지표 — 3계층 구조

본 실험이 기존 실험과 근본적으로 다른 점은 **Teacher-level, Logit-level, Student-level 세 층위를 동시에 측정**한다는 것이다. 각 층위는 2.1, 2.2 절의 주장 중 어느 하나와 대응된다.

**Layer 1: Teacher-level (가정 A 검증용)**

| 지표 | 계산 방법 | 해석 |
|---|---|---|
| Per-class F1 (K × C 행렬) | 각 Teacher가 proxy test set에서 클래스별로 계산 | 개별 Teacher의 분류 능력 |
| Teacher의 proxy 상 global accuracy | Teacher별 전체 proxy accuracy | Teacher 전체 성능 |
| Coverage count per class | F1 > 0.3인 Teacher 수 | "이 클래스를 아는 Teacher가 몇 명인가" |

**Layer 2: Logit-level (가정 B 검증용) — 신규 추가 지표**

| 지표 | 계산 방법 | 해석 |
|---|---|---|
| 평균 logit entropy | 각 Teacher의 proxy logit entropy 평균 | 높을수록 dark knowledge 풍부 |
| Top-1 softmax confidence 분포 | 전체 proxy 샘플의 top-1 softmax 값 히스토그램 | 1에 수렴할수록 hard-label화 |
| Top-2 gap 분포 | (top-1 - top-2) softmax 값 히스토그램 | 크면 hard-label화, 작으면 soft |
| 전문/비전문 클래스별 entropy 비교 | 각 Teacher의 전문 클래스(F1>0.5) vs 비전문 클래스 entropy | 각 영역의 품질 개별 확인 |

**이 Layer 2가 본 실험의 가장 중요한 신규 기여다.** 기존 v4/v5 실험은 Layer 1과 Layer 3만 측정했으며, "왜" KD가 실패하는지에 대한 직접 증거가 부족했다.

**Layer 3: Student-level (최종 결과)**

| 지표 | 계산 방법 | 해석 |
|---|---|---|
| Global test accuracy | Balanced test set (각 클래스 균등) | 최종 KD 결과 |
| Macro F1 | 클래스별 F1의 평균 | Global acc의 보조 |
| Gap Recovery (%) | (KD acc - Lower) / (Upper - Lower) × 100 | KD의 기여도 (핵심 비교 metric) |
| Per-class accuracy | 각 클래스별 Student acc | Worst-K / Best-K 분석용 |

### 3.5 예상 결과 패턴 (가설)

**H1 (Teacher F1의 단조 증가)**: α가 감소할수록 개별 Teacher의 per-class F1 (전문 클래스에서)은 **증가**한다. 이유: 집중도가 올라가면 해당 클래스에 대한 학습 샘플 수가 늘어나고, 다른 클래스와의 경쟁이 줄어든다.

**H2 (Logit entropy의 단조 감소)**: α가 감소할수록 Teacher logit의 평균 entropy는 **감소**한다. 특히 전문 클래스 logit의 top-1 confidence가 1에 수렴한다 (hard-label화).

**H3 (Student accuracy의 단조 감소, 핵심)**: α가 감소할수록 Student global accuracy는 **감소**한다. 가중치 방법(Uniform / Top-1 / Top-3)과 무관하게 이 경향이 유지된다.

**H4 (Top-K의 역설)**: α=0.1에서 Top-1이 Uniform보다 **낮은** 성능을 보인다. 이유: Top-1은 hard-label화된 logit 하나만 쓰므로 dark knowledge가 거의 0이고, Uniform은 여러 Teacher를 평균하여 최소한의 noise cancellation 효과가 있다. 반대로 α=100(IID)에서는 Top-K나 Uniform이 거의 동일한 성능을 보인다 (Teacher 간 차이가 없으므로 선별이 무의미).

**H5 (IID 단일 Teacher KD와의 격차)**: 모든 Non-IID 조건의 Student 성능은 **single-teacher IID KD (~0.686)보다 낮다**. 이것이 가장 강력한 증거 — "데이터를 잘게 쪼개서 학습한 여러 Teacher의 logit을 모아도, 한 곳에 모아 학습한 한 Teacher의 logit 하나를 못 이긴다."

**핵심 그래프 (예상)**: 2축 그래프

```
x축: α (log scale, 0.1 → 100)
y축 (왼쪽): 평균 Teacher per-class F1 — 전문 클래스에서 측정
y축 (오른쪽): Student global accuracy

Teacher F1 곡선:  α↓일수록 상승
Student Acc 곡선: α↓일수록 하락
               → 두 곡선이 명확히 역상관
```

이 그래프 하나가 "가정 A가 잘못되었다"는 것을 시각적으로 증명한다.

---

## 4. 시각화 체크리스트

기존 `eacs_kd_multi_teacher.py`는 `plot_client_distribution`과 `plot_expertise_heatmap`을 이미 제공한다. 본 실험에서는 이에 더해 다음 시각화를 **모든 α 조건에서** 생성한다.

### 4.1 파티션 시각화 (실험 조건 확인용)

**Figure 1a. Client × Class 데이터 할당 히트맵 (α별 subplot)**
- 5개 α 값에 대해 각각 (K=5 × C=100) 히트맵을 subplot으로 배치
- 한 장에 5개 패널로 표시하여 α 변화에 따른 분배 극단화를 한눈에
- 셀 값 = log(데이터 수 + 1), 컬러맵: viridis

**Figure 1b. Client별 클래스 수 통계**
- x축: α, y축: "F1 측정 가능한 클래스 수" (샘플 > 0인 클래스)
- K명의 Teacher의 평균과 표준편차를 error bar로
- α가 낮아질수록 Teacher별 도메인이 좁아지는 것을 정량화

**Figure 1c. Client별 데이터 총량 (Stacked bar)**
- α별로 각 Teacher가 받은 데이터 총량 + top-3 클래스 구성을 stacked bar로
- 총량은 α와 무관하게 균등해야 함을 확인 (sanity check)

### 4.2 Teacher 성능 시각화 (Layer 1)

**Figure 2a. Expertise Matrix 히트맵 (α별 subplot, 5개)**
- (K × C) F1 행렬을 α별로 비교
- α=100: 거의 균등한 값 (모든 Teacher가 모든 클래스 비슷)
- α=0.1: sparse한 행렬 (각 Teacher의 전문 영역만 밝게)

**Figure 2b. Teacher 성능 박스플롯**
- x축: α, y축: 각 Teacher의 proxy global accuracy
- 박스플롯으로 분포 표시 (K명의 Teacher 간 분산)
- α↓일수록 분산 증가 예상

### 4.3 Logit 품질 시각화 (Layer 2, 신규)

**Figure 3a. Logit Entropy 분포 (α별 violin plot)**
- x축: α, y축: 샘플당 logit entropy
- α=100: 높고 좁은 분포 (다양성 풍부)
- α=0.1: 낮고 넓은 분포 (전문/비전문 양극화)

**Figure 3b. Top-1 Confidence 누적 분포 (CDF)**
- x축: top-1 softmax 값, y축: 샘플 비율
- α별로 다른 곡선
- α=0.1에서 우측(1.0 근처)에 집중된 곡선 기대 (hard-label화)

**Figure 3c. 전문/비전문 클래스별 Entropy 비교**
- x축: α, y축: entropy
- 두 곡선: "전문 클래스에서의 entropy" vs "비전문 클래스에서의 entropy"
- α=0.1에서 둘 다 낮음 (전자는 sharp, 후자는 random이 아닌 majority bias)
- 가정 B 반박의 직접 증거

### 4.4 Student 결과 시각화 (Layer 3)

**Figure 4a. 핵심 역상관 그래프 (본 보고서의 main figure)**
- 2축 그래프: x=α(log), 왼쪽 y=Teacher F1, 오른쪽 y=Student Acc
- 각 점마다 3-seed 평균 ± std
- Single-Teacher IID KD(~0.686)를 수평선으로 overlay

**Figure 4b. Weighting별 Student Acc 비교**
- x축: α, y축: Student Global Acc
- 3개 곡선: Uniform / Top-1 / Top-3
- 수평선으로 Lower Bound, Upper Bound, Single-Teacher KD 기준선 표시
- α↓에서 Top-1이 Uniform 이하로 떨어지는지 확인 (H4)

**Figure 4c. Gap Recovery 곡선**
- x축: α, y축: Gap Recovery (%)
- 3개 곡선 + 0% 기준선
- 음의 영역에 들어가는 조건이 어디인지 명시

**Figure 4d. Per-class accuracy 비교 (α=0.1 대표 조건)**
- x축: 클래스 (Lower Bound 기준 정렬)
- y축: Student accuracy
- 3개 weighting 방법을 다른 색으로

### 4.5 시각화 구현상 보강 사항

현재 `eacs_kd_multi_teacher.py`는 단일 α로만 실행되므로, 본 실험을 위해 다음이 필요하다:

1. **α 스윕 실행 스크립트** (`run_alpha_sweep.sh`): 5개 α × 3 seed를 순차 실행, 결과를 구조화된 디렉토리에 저장
2. **결과 집계 스크립트** (`aggregate_sweep.py`): 각 run의 결과 json/npz를 읽어 통합 시각화 생성
3. **Layer 2 지표 계산 코드 추가**: 현재 스크립트에는 logit entropy/confidence 분포 계산이 없음 → Teacher logit collection 직후 추가

---

## 5. 연구 결과물의 예상 활용

본 실험 결과는 연구 방향에 따라 두 가지로 활용 가능하다.

### 5.1 방향 A — Negative Result 논문의 핵심 증거

`critical_review_v2`의 방향 A(Negative result + 분석 논문)로 가는 경우, 본 실험 결과는 논문의 핵심 empirical evidence가 된다. 논문 구조는 다음과 같이 예상된다:

- Section 3 (Problem Analysis): 가정 A, B의 이론적 분석 (critical_review 내용 활용)
- Section 4 (Experiments): 본 실험 결과 — Figure 4a (핵심 역상관) + Figure 3 (logit 품질) + Figure 4b-c (Gap Recovery)
- Section 5 (Discussion): 기존 FL-KD 논문들의 가정 재검토

### 5.2 방향 B — 새로운 연구 방향의 motivation

`critical_review_v2`의 방향 B, C, D(multi-round co-distillation, drift 시뮬레이션, 문제 재정의)로 가는 경우, 본 실험 결과는 **왜 기존 접근이 부족한지**를 보여주는 motivation 실험이 된다.

어느 방향이든 본 실험은 선행되어야 하며, 결과 자체가 독립적 가치를 가진다.

---

## 6. 구현 및 일정

### 6.1 코드 변경 사항 (기존 `eacs_kd_multi_teacher.py` 기반)

| 영역 | 변경 내용 | 우선순위 |
|---|---|---|
| 파티션 | 변경 없음 (기존 유지) | — |
| Teacher 학습 | 변경 없음 | — |
| Logit 수집 | **Layer 2 지표 계산 추가** (entropy, top-1, top-2 gap) | 높음 |
| Weighting | Uniform, Top-1, Top-3만 유지 (Piecewise 제외) | 중간 |
| 시각화 | **α 스윕 결과 집계용 `aggregate_sweep.py` 신규** | 높음 |
| 실행 | `run_alpha_sweep.sh` 신규 | 높음 |

### 6.2 예상 GPU 시간 (g5.2xlarge 기준)

| 작업 | 단위 시간 | 횟수 | 총 시간 |
|---|---|---|---|
| Teacher 학습 (K=5) | ~2h | 5 α × 3 seed = 15 | ~30h |
| Lower Bound | ~1h | 3 seed | ~3h |
| Upper Bound | ~4h | 3 seed | ~12h |
| KD 학습 | ~1.5h | 5 α × 3 weighting × 3 seed = 45 | ~68h |
| **총 예상** | | | **~113h (약 5일)** |

Teacher 학습 결과는 weighting 방법과 무관하므로 15회만 돌리면 됨(Teacher 캐싱 활용). Upper Bound도 α와 무관하므로 3회로 충분.

### 6.3 실행 계획

```
1주차: 코드 보강 (Layer 2 지표 + 집계 스크립트)
2주차 초: Bounds + 첫 α (α=0.1) 3-seed 실행으로 파이프라인 검증
2주차 후반 ~ 3주차: 나머지 α 값들 순차 실행
3주차 후반: 집계 및 시각화, 본 보고서 결과 섹션 업데이트
```

---

## 7. 교수님께 드리는 질의

본 실험 설계 전에 확인 부탁드리는 사항들입니다:

1. **Scope 결정**: 본 실험을 "negative result 자체를 보이는 것"으로 정리할지, 아니면 이 결과를 발판으로 삼아 **대안 방향(multi-round, drift 등)**으로 바로 넘어갈지에 대한 의견을 듣고자 합니다. 전자는 논문 하나로 마무리될 수 있고, 후자는 본 실험이 예비 실험(pilot)으로 축소됩니다.

2. **비교군 충실도**: 본 실험은 의도적으로 EaCS-KD 계열 가중치(Uniform, Top-K) 간 비교에 집중하고 있습니다. FedAvg, FedDF 같은 외부 baseline을 포함시키는 것이 "negative result의 보편성"을 강화하는 데 필수인지, 아니면 scope 확장이 오히려 메시지를 흐리는지에 대한 판단을 구합니다.

3. **Layer 2 지표의 중요도**: logit entropy와 top-1 confidence 분포는 "왜 안 되는가"를 직접 보이는 핵심 지표지만, 기존 FL-KD 논문들에서는 거의 측정되지 않습니다. 이 지표들을 신규 contribution으로 강조해도 좋을지, 아니면 보조 분석 정도로 남겨두는 것이 안전할지 의견을 듣고자 합니다.

4. **α=100 조건의 처리**: α=100은 사실상 IID에 가깝고, 이 조건에서 Teacher를 5명으로 나누는 것은 "데이터 양이 1/5로 준 IID 학습"이 됩니다. 이 조건의 결과를 single-teacher IID KD와 직접 비교할지, 아니면 **IID 조건의 multi-client는 단순히 "noise cancellation이 극대화된 상태"**로 해석할지에 대한 관점 정리가 필요합니다.

---

## Appendix A. 기존 결과와의 연결

| 기존 관찰 | 본 실험이 보강하는 것 |
|---|---|
| v4 regression (Naive KD < Lower, α=0.1) | α 스윕으로 regression이 발생하는 조건의 경계를 정량화 |
| v5의 Uniform > Top-K F1 (α=0.1) | 가중치 방법과 α의 상호작용을 체계적으로 매핑 |
| Single-Teacher KD ~0.686 | 모든 Non-IID multi-teacher 조건이 이 기준선 아래임을 보임 (H5) |
| critical_review_v2의 이론적 분석 | Layer 2 지표로 이론 주장을 실측으로 뒷받침 |

## Appendix B. 제외된 설계 옵션과 그 이유

| 옵션 | 제외 이유 |
|---|---|
| Piecewise weighting 포함 | 본 실험의 메시지(구조적 한계)와 독립된 질문(가중치 우열). 별도 실험에서 다룸 |
| Heterogeneous Teacher 아키텍처 | 변수 축 추가로 해석 복잡화. 본 실험은 동일 구조(ResNet-18)로 고정 |
| Pretrained backbone | v3_2에서 이미 문제 확인됨. 본 실험은 scratch로 통일 |
| Proxy size 스윕 | 주 변수가 아님. 필요 시 α=0.5 대표 조건에서만 ablation |
| CIFAR-100 병행 | 현재 파이프라인이 ImageNet-100에 최적화. 시간 절약을 위해 단일 데이터셋 |
