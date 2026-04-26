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