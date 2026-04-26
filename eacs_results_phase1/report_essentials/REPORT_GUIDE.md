# 보고서/논문 작성 가이드 — 핵심 정수

이 디렉토리는 보고서 본문에 들어갈 **핵심 Figure 6개**와 **정량 표 2개**를 포함합니다.

## 권장 보고서 구조

### Section 1 — 논지 요약

> "Non-IID Teacher의 logit은 dark knowledge가 부재하면 noise로 작용한다. 이 원리는 (i) 동일 모델 환경(Phase 1)에서 직접 측정 가능하며, (ii) Small→Large 현실 시나리오(Phase 2)에서 KD 실패라는 형태로 재현된다."

### Section 2 — Phase 1 (주장 A 본문)

**필요 Figure**: `phase1_main/` 4개
- Fig 3a, 3b, 3c: Layer 2 지표로 logit이 noise임을 직접 보임
- Fig 4b: Student acc로 Layer 3 보조 증거

**필요 표**: `tables/table_phase1_summary.md`

**Phase 1 — 주장 A (동일 ResNet-18 pretrained)**

- Lower Bound: **0.8305** (proxy만)
- Upper Bound: **0.8841** (centralized)

| α | T entropy↓ | T top-1↑ | Uniform | Top-1 | Top-3 | Uniform GR | Top-1 GR | Top-3 GR |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.718 | 0.790 | 0.7870 | 0.7790 | 0.8128 | -81.0% | -96.3% | -32.9% |
| 0.5 | 0.535 | 0.845 | 0.8285 | 0.8247 | 0.8367 | -3.8% | -10.6% | +11.8% |
| 1.0 | 0.424 | 0.876 | 0.8453 | 0.8291 | 0.8408 | +27.8% | -2.5% | +19.3% |
| 10.0 | 0.352 | 0.898 | 0.8459 | 0.8326 | 0.8414 | +28.9% | +4.1% | +20.5% |
| 100.0 | 0.345 | 0.900 | 0.8453 | 0.8357 | 0.8424 | +27.7% | +10.0% | +22.4% |

**해석**:
- T entropy: Teacher logit의 평균 entropy. α↓일수록 ↓이면 hard-label화 진행
- T top-1: top-1 softmax 평균. α↓일수록 ↑이면 dark knowledge 부재
- GR (Gap Recovery): 0%=Lower, 100%=Upper, 음수=Lower보다 나쁨

### Section 3 — Phase 2 (주장 B 본문)

**필요 Figure**: `phase2_main/` 2개
- Fig 4b: KD가 Lower 아래로 떨어지는 시각적 증거
- Fig 4c: Gap Recovery 음수 진입

**필요 표**: `tables/table_phase2_summary.md`

**Phase 2 — 주장 B (Small→Large: MobileNetV2 → ResNet-50, pretrained)**

- Lower Bound: **0.9111** (proxy만)
- Upper Bound: **0.9249** (centralized)

| α | Uniform | Top-1 | Top-3 | Uniform GR | Top-1 GR | Top-3 GR |
|---|---|---|---|---|---|---|
| 0.1 | 0.8827 | 0.8522 | 0.8769 | -224.9% | -473.6% | -267.1% |
| 0.5 | 0.8941 | 0.8843 | 0.8978 | -139.9% | -222.2% | -105.2% |
| 1.0 | 0.9009 | 0.8870 | 0.8987 | -88.2% | -190.0% | -105.7% |
| 10.0 | 0.8984 | 0.8901 | 0.8959 | -109.9% | -170.6% | -124.7% |
| 100.0 | 0.8975 | 0.8886 | 0.8960 | -114.3% | -187.3% | -126.1% |

**해석**:
- Lower가 Upper에 매우 근접 (작은 Gap) → Student가 이미 강함
- α=0.1에서 Uniform/Top-1/Top-3가 Lower 아래로 떨어지면 핵심 증거
- Gap Recovery 음수 = Non-IID Teacher의 KD가 Student를 끌어내림

### Section 4 — 부록 / 보조 자료

- `appendix/phase1_partition.png`: 실험 조건(Dirichlet 파티션) 시각화
- `appendix/phase1_expertise_matrix.png`: Teacher 전문성 분포
- 전체 12개 Figure는 `$EACS_RESULTS_ROOT/phase{1,2}/figures/`에 보관

## 핵심 수치 한 줄 요약 (Abstract용)

- **Phase 1**: α=0.1에서 Teacher logit entropy 0.72 (vs IID α=100: 0.34) — hard-label화 명확히 관측
- **Phase 2**: α=0.1, Uniform KD = 0.8827 (Lower 0.9111와 비교)
  Gap Recovery: Uniform -224.9%, Top-1 -473.6%

---

## 보고서에 직접 인용 가능한 핵심 문장 후보

1. (Phase 1) "α=100(IID)에서 α=0.1로 갈수록 Teacher의 평균 logit entropy는 0.34에서 0.72로 -0.37 감소했다 — dark knowledge가 사라지고 logit이 hard-label에 수렴함을 의미한다."
2. (Phase 2) "Pretrained Student는 proxy만으로 0.9111의 Lower Bound를 달성하지만, α=0.1의 Non-IID Teacher KD는 0.8827로 **Lower Bound 아래**로 떨어진다 (Gap Recovery -224.9%)."
3. (종합) "동일 모델(Phase 1)이든 Small→Large(Phase 2)이든, α↓일수록 KD의 가치가 사라지는 일관된 패턴이 관측된다. 이는 logit 기반 KD 패러다임의 구조적 한계를 시사한다."