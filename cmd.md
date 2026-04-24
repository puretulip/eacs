네, Teacher 학습과 KD 학습에 필요한 명령어를 정리해드리겠습니다.

## Step 3: Teacher 학습 (10개 명령)

α 5개 × seed 2개 = 10개 조합입니다.

```bash
# seed=42
python train_teachers.py --alpha 0.1   --seed 42
python train_teachers.py --alpha 0.5   --seed 42
python train_teachers.py --alpha 1.0   --seed 42
python train_teachers.py --alpha 10.0  --seed 42
python train_teachers.py --alpha 100.0 --seed 42

# seed=123
python train_teachers.py --alpha 0.1   --seed 123
python train_teachers.py --alpha 0.5   --seed 123
python train_teachers.py --alpha 1.0   --seed 123
python train_teachers.py --alpha 10.0  --seed 123
python train_teachers.py --alpha 100.0 --seed 123
```

## Step 4: KD 학습 (30개 명령)

α 5개 × seed 2개 × weighting 3개 = 30개 조합입니다.

```bash
# =====================================
# seed=42
# =====================================
# α=0.1
python run_kd.py --alpha 0.1 --seed 42 --weighting uniform
python run_kd.py --alpha 0.1 --seed 42 --weighting top_1
python run_kd.py --alpha 0.1 --seed 42 --weighting top_3

# α=0.5
python run_kd.py --alpha 0.5 --seed 42 --weighting uniform
python run_kd.py --alpha 0.5 --seed 42 --weighting top_1
python run_kd.py --alpha 0.5 --seed 42 --weighting top_3

# α=1.0
python run_kd.py --alpha 1.0 --seed 42 --weighting uniform
python run_kd.py --alpha 1.0 --seed 42 --weighting top_1
python run_kd.py --alpha 1.0 --seed 42 --weighting top_3

# α=10.0
python run_kd.py --alpha 10.0 --seed 42 --weighting uniform
python run_kd.py --alpha 10.0 --seed 42 --weighting top_1
python run_kd.py --alpha 10.0 --seed 42 --weighting top_3

# α=100.0
python run_kd.py --alpha 100.0 --seed 42 --weighting uniform
python run_kd.py --alpha 100.0 --seed 42 --weighting top_1
python run_kd.py --alpha 100.0 --seed 42 --weighting top_3

# =====================================
# seed=123
# =====================================
# α=0.1
python run_kd.py --alpha 0.1 --seed 123 --weighting uniform
python run_kd.py --alpha 0.1 --seed 123 --weighting top_1
python run_kd.py --alpha 0.1 --seed 123 --weighting top_3

# α=0.5
python run_kd.py --alpha 0.5 --seed 123 --weighting uniform
python run_kd.py --alpha 0.5 --seed 123 --weighting top_1
python run_kd.py --alpha 0.5 --seed 123 --weighting top_3

# α=1.0
python run_kd.py --alpha 1.0 --seed 123 --weighting uniform
python run_kd.py --alpha 1.0 --seed 123 --weighting top_1
python run_kd.py --alpha 1.0 --seed 123 --weighting top_3

# α=10.0
python run_kd.py --alpha 10.0 --seed 123 --weighting uniform
python run_kd.py --alpha 10.0 --seed 123 --weighting top_1
python run_kd.py --alpha 10.0 --seed 123 --weighting top_3

# α=100.0
python run_kd.py --alpha 100.0 --seed 123 --weighting uniform
python run_kd.py --alpha 100.0 --seed 123 --weighting top_1
python run_kd.py --alpha 100.0 --seed 123 --weighting top_3
```

## 실전 실행 방법 (EC2별 묶음)

위 명령어들을 하나씩 수동으로 실행하는 것보다는, **EC2 한 대가 자기 몫을 논스톱으로 돌리도록** 묶어서 실행하는 게 낫습니다. 아래는 5대 EC2에 분산하는 예시입니다.

### EC2 #A — α=0.1 전담 (seed 2개 × weighting 3개 = Teacher 2 + KD 6)

```bash
nohup bash -c '
set -e
# Teacher 먼저
python train_teachers.py --alpha 0.1 --seed 42
python train_teachers.py --alpha 0.1 --seed 123

# KD 3개씩 × 2 seed
for seed in 42 123; do
  for w in uniform top_1 top_3; do
    python run_kd.py --alpha 0.1 --seed $seed --weighting $w
  done
done
sudo shutdown -h now
' > ec2_A_alpha01.log 2>&1 &
```

### EC2 #B — α=0.5 전담

```bash
nohup bash -c '
set -e
python train_teachers.py --alpha 0.5 --seed 42
python train_teachers.py --alpha 0.5 --seed 123
for seed in 42 123; do
  for w in uniform top_1 top_3; do
    python run_kd.py --alpha 0.5 --seed $seed --weighting $w
  done
done
sudo shutdown -h now
' > ec2_B_alpha05.log 2>&1 &
```

### EC2 #C — α=1.0 전담

```bash
nohup bash -c '
set -e
python train_teachers.py --alpha 1.0 --seed 42
python train_teachers.py --alpha 1.0 --seed 123
for seed in 42 123; do
  for w in uniform top_1 top_3; do
    python run_kd.py --alpha 1.0 --seed $seed --weighting $w
  done
done
sudo shutdown -h now
' > ec2_C_alpha1.log 2>&1 &
```

### EC2 #D — α=10.0 전담

```bash
nohup bash -c '
set -e
python train_teachers.py --alpha 10.0 --seed 42
python train_teachers.py --alpha 10.0 --seed 123
for seed in 42 123; do
  for w in uniform top_1 top_3; do
    python run_kd.py --alpha 10.0 --seed $seed --weighting $w
  done
done
sudo shutdown -h now
' > ec2_D_alpha10.log 2>&1 &
```

### EC2 #E — α=100.0 전담

```bash
nohup bash -c '
set -e
python train_teachers.py --alpha 100.0 --seed 42
python train_teachers.py --alpha 100.0 --seed 123
for seed in 42 123; do
  for w in uniform top_1 top_3; do
    python run_kd.py --alpha 100.0 --seed $seed --weighting $w
  done
done
sudo shutdown -h now
' > ec2_E_alpha100.log 2>&1 &
```

## 주의사항 3가지

**1. 선행 조건 확인**
각 EC2에서 돌리기 전에, `prepare_partition.py --all`이 이미 실행되어 EFS에 파티션 `.npz` 파일들이 존재해야 합니다. 또한 `run_kd.py`는 **Gap Recovery 계산을 위해 Bounds 결과를 참조**하지만, Bounds가 없어도 KD 자체는 문제없이 돌아갑니다 (gap_recovery만 null로 저장). 가능하면 Bounds 학습을 먼저 끝낸 뒤 KD를 돌리는 것이 좋습니다.

**2. `set -e`의 의미**
위 스크립트의 `set -e`는 "중간에 명령 하나라도 실패하면 즉시 중단"이란 뜻입니다. 셧다운이 일어나지 않으니 수동으로 상황을 확인할 수 있습니다. 반대로 **"무슨 일이 있어도 EC2는 끄고 싶다"** 면 `set -e`를 빼고 마지막 줄에 `sudo shutdown` 대신 `; sudo shutdown -h now`를 조건부로 붙이지 말고 **무조건 실행**하는 형태로 쓰세요:

```bash
nohup bash -c '(...)  ; sudo shutdown -h now' > log 2>&1 &
```

**3. 병렬 실행 시 `teachers.pt` 경합 없음**
각 EC2는 서로 다른 `(α, seed)` 조합만 담당하므로 저장 디렉토리도 다릅니다 (`teachers/alpha0.1_seed42/` vs `teachers/alpha0.5_seed42/`). 충돌 위험 없습니다.

## 진행 상황 모니터링

다른 터미널에서:

```bash
# 로그 실시간 확인
tail -f ec2_A_alpha01.log

# 완료된 Teacher 목록
ls $EACS_RESULTS_ROOT/teachers/

# 완료된 KD 목록
ls $EACS_RESULTS_ROOT/kd_runs/

# 에폭 시간 로그
tail -f $EACS_RESULTS_ROOT/logs/teacher_a0.1_s42_k0.log
```

모든 EC2가 끝난 뒤에는 **한 대에서** 다음 명령으로 집계:

```bash
python aggregate_sweep.py
```

추가로 궁금한 점 있으신가요?
