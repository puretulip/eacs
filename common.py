"""
common.py — 공통 유틸리티 모듈
=====================================

모든 스크립트가 공유하는 구성 요소:
  - 디렉토리 경로 규약 (EACS_RESULTS_ROOT, EACS_DATA_ROOT)
  - ParquetImageDataset (HuggingFace ImageNet-100 parquet 포맷)
  - 모델 생성 (ResNet-18 scratch)
  - Dirichlet 파티션
  - 평가 지표 계산
  - 재현성 유틸 (seed 고정)
  - 에폭 시간 로거
"""

import os
import io
import json
import time
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as tvm
from PIL import Image


# =============================================================================
# 경로 규약
# =============================================================================
# 환경변수로 results_root를 지정. EFS 경로로 설정하면 EC2 간 공유 가능.
RESULTS_ROOT = Path(os.environ.get(
    "EACS_RESULTS_ROOT",
    os.path.expanduser("~/eacs_results")
))
# DATA_ROOT 아래에 train.parquet, validation.parquet, class_mapping.txt 존재.
# 기존 HuggingFace ImageNet-100 구조 (eacs_kd_multi_teacher.py와 동일).
DATA_ROOT = Path(os.environ.get(
    "EACS_DATA_ROOT",
    os.path.expanduser("~/data/imagenet100")
))
TRAIN_PARQUET = DATA_ROOT / "train.parquet"
VAL_PARQUET = DATA_ROOT / "validation.parquet"
CLASS_MAPPING = DATA_ROOT / "class_mapping.txt"


def partition_path(alpha, seed):
    # 파티션은 Phase와 무관 (두 Phase가 동일 파티션 공유)
    return RESULTS_ROOT / "partitions" / f"alpha{alpha}_seed{seed}.npz"

def bounds_dir(seed, phase=1):
    return RESULTS_ROOT / f"phase{phase}" / "bounds" / f"seed{seed}"

def teachers_dir(alpha, seed, phase=1):
    return RESULTS_ROOT / f"phase{phase}" / "teachers" / f"alpha{alpha}_seed{seed}"

def kd_dir(alpha, seed, weighting, phase=1):
    return RESULTS_ROOT / f"phase{phase}" / "kd_runs" / f"alpha{alpha}_seed{seed}_{weighting}"

def logs_dir(phase=None):
    if phase is None:
        return RESULTS_ROOT / "logs"
    return RESULTS_ROOT / f"phase{phase}" / "logs"


def ensure_dirs(phase=None):
    """결과 디렉토리 구조 생성.
    phase=None: 파티션/공통 디렉토리만
    phase=1 or 2: 해당 phase 디렉토리 포함
    """
    (RESULTS_ROOT / "partitions").mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    if phase is not None:
        for sub in ["bounds", "teachers", "kd_runs", "logs", "figures"]:
            (RESULTS_ROOT / f"phase{phase}" / sub).mkdir(parents=True, exist_ok=True)


# =============================================================================
# 재현성
# =============================================================================
def set_seed(seed: int, deterministic: bool = False):
    """재현성을 위한 시드 고정.

    deterministic=True면 cudnn benchmark 비활성화 (느려지지만 완전 재현).
    학습 실험에서는 보통 False (속도 우선).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# =============================================================================
# 시간 로거
# =============================================================================
class EpochTimer:
    """에폭별 소요 시간을 기록하고 로그로 남김.

    Usage:
        timer = EpochTimer(tag="teacher_k0", log_path=path)
        for epoch in range(E):
            with timer.epoch(epoch):
                train_one_epoch(...)
        timer.summary()
    """
    def __init__(self, tag: str, log_path: Path = None):
        self.tag = tag
        self.log_path = Path(log_path) if log_path else None
        self.records = []
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def epoch(self, epoch_idx: int):
        start = time.time()
        yield
        elapsed = time.time() - start
        rec = {"epoch": epoch_idx, "elapsed_sec": elapsed}
        self.records.append(rec)
        msg = f"[{self.tag}] epoch {epoch_idx:3d} | {elapsed:7.1f}s"
        print(msg, flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"tag": self.tag, **rec}) + "\n")

    def summary(self):
        if not self.records:
            return {}
        elapsed = [r["elapsed_sec"] for r in self.records]
        s = {
            "tag": self.tag,
            "n_epochs": len(elapsed),
            "total_sec": sum(elapsed),
            "mean_sec": float(np.mean(elapsed)),
            "min_sec": float(np.min(elapsed)),
            "max_sec": float(np.max(elapsed)),
        }
        print(f"[{self.tag}] SUMMARY | n={s['n_epochs']} "
              f"total={s['total_sec']:.0f}s "
              f"mean={s['mean_sec']:.1f}s "
              f"(min={s['min_sec']:.1f}, max={s['max_sec']:.1f})",
              flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"SUMMARY": s}) + "\n")
        return s


# =============================================================================
# 데이터셋 — ImageNet-100
# =============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_transforms(train: bool, img_size: int = 224):
    """ImageNet 표준 augmentation."""
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 256 / 224)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class ParquetImageDataset(Dataset):
    """Parquet 기반 ImageNet-100 데이터셋.

    Parquet에는 두 컬럼이 있다고 가정:
      - 'image': bytes (또는 {'bytes': ...} dict) — JPEG/PNG encoded
      - 'label': int — 클래스 인덱스 (0..99)

    HuggingFace datasets로 to_parquet 된 파일 포맷을 지원.
    기존 eacs_kd_multi_teacher.py의 ParquetImageDataset과 동일 동작.
    """
    def __init__(self, parquet_path=None, transform=None, shared_table=None):
        """shared_table이 주어지면 이미 로드된 pyarrow.Table을 재사용.

        같은 parquet을 train/noaug 두 dataset으로 쓸 때 메모리 절약 가능.
        """
        import pyarrow.parquet as pq
        if shared_table is not None:
            self.table = shared_table
        else:
            if parquet_path is None:
                raise ValueError("parquet_path 또는 shared_table이 필요합니다")
            print(f"  Loading parquet: {parquet_path}")
            self.table = pq.read_table(str(parquet_path))
        # 라벨은 한 번에 파이썬 리스트로 — targets 호환을 위해
        self.labels = self.table.column("label").to_pylist()
        # image 컬럼은 lazy access (메모리 절약)
        self.image_col = self.table.column("image")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_data = self.image_col[idx].as_py()
        # HuggingFace format: {'bytes': ..., 'path': ...} 또는 raw bytes
        if isinstance(img_data, dict):
            img_bytes = img_data["bytes"]
        else:
            img_bytes = img_data
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]

    @property
    def targets(self):
        """torchvision 호환 — dirichlet_partition 등에서 사용."""
        return self.labels


def load_parquet_table(split: str):
    """split 'train' 또는 'val' 해당 parquet 테이블을 로드.
    반환된 테이블은 여러 Dataset 객체에서 shared_table로 공유 가능.
    """
    import pyarrow.parquet as pq
    if split == "train":
        path = TRAIN_PARQUET
    elif split in ("val", "validation"):
        path = VAL_PARQUET
    else:
        raise ValueError(f"Unknown split: {split}")
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet 파일 없음: {path}\n"
            f"EACS_DATA_ROOT 환경변수 확인 (현재: {DATA_ROOT})\n"
            f"기대 구조: $EACS_DATA_ROOT/{{train,validation}}.parquet"
        )
    print(f"  Loading parquet: {path}")
    return pq.read_table(str(path))


def load_class_mapping():
    """class_mapping.txt를 딕셔너리로 반환 (없어도 에러 아님, None 반환)."""
    if not CLASS_MAPPING.exists():
        return None
    mapping = {}
    with open(CLASS_MAPPING) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "0 n01440764 tench" 또는 "n01440764 tench" 형식 모두 허용
            parts = line.split(maxsplit=2)
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    mapping[idx] = " ".join(parts[1:])
                except ValueError:
                    mapping[len(mapping)] = line
    return mapping


class IndexedSubset(Dataset):
    """Subset + (image, label, position_in_subset) 반환.

    KD에서 Teacher logit을 사전 수집 후 DataLoader shuffle 하에서
    정확한 logit 매핑을 위해 원 위치 인덱스를 함께 반환.
    """
    def __init__(self, base_dataset: Dataset, indices: np.ndarray):
        self.base = base_dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[int(self.indices[i])]
        return img, label, i  # i = proxy 내 위치 (Teacher logit 인덱스와 동일)


# =============================================================================
# Dirichlet 파티션
# =============================================================================
def dirichlet_partition(labels: np.ndarray, num_clients: int,
                        alpha: float, num_classes: int,
                        seed: int, min_samples_per_client: int = 10):
    """클래스별 Dirichlet(α) 비율로 샘플을 K명 클라이언트에 분배.

    표준 Non-IID 시뮬레이션 방식 (McMahan 2017 기반).
    클래스 c의 샘플을 (p_1, p_2, ..., p_K) ~ Dir(α)로 K명에게 비례 분배.

    alpha가 크면 균등(IID), 작으면 극단 집중(Non-IID).

    Returns:
      client_indices: {k: np.ndarray} — 각 클라이언트에게 할당된 (labels 배열 내) 인덱스
    """
    rng = np.random.RandomState(seed)
    # 각 클라이언트의 (인덱스 리스트) 초기화
    client_indices = {k: [] for k in range(num_clients)}

    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        # Dirichlet 샘플 — K개 비율
        proportions = rng.dirichlet([alpha] * num_clients)
        # 비율 → 경계
        splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        parts = np.split(idx_c, splits)
        for k in range(num_clients):
            client_indices[k].extend(parts[k].tolist())

    # shuffle 및 array 변환
    for k in range(num_clients):
        arr = np.array(client_indices[k])
        rng.shuffle(arr)
        client_indices[k] = arr

    # 최소 샘플 수 보장 (너무 적으면 학습 불가)
    for k in range(num_clients):
        if len(client_indices[k]) < min_samples_per_client:
            # 이런 경우는 α가 매우 낮고 seed가 극단적일 때 발생
            # 경고만 출력, 학습은 진행
            print(f"  [WARN] client {k}: only {len(client_indices[k])} samples",
                  flush=True)

    return client_indices


# =============================================================================
# 모델
# =============================================================================
def build_resnet18(num_classes: int = 100, pretrained: bool = True):
    """ResNet-18 (11M) — Phase 1 공용 모델.

    Phase 1: Teacher = Student = Bounds = ResNet-18 (pretrained)
    주장 A 증명용: 동일 모델로 변수 통제, logit이 noise인지를 Layer 2로 증명.
    """
    if pretrained:
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = tvm.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenet_v2(num_classes: int = 100, pretrained: bool = True):
    """MobileNetV2 (3.4M) — Phase 2 Teacher용.

    Phase 2: Small Teacher → Large Student 현실 시나리오.
    ImageNet-1K pretrained, Non-IID 데이터로 full fine-tuning.
    """
    if pretrained:
        model = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = tvm.mobilenet_v2(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet50(num_classes: int = 100, pretrained: bool = True):
    """ResNet-50 (25M) — Phase 2 Student / Bounds용.

    Phase 2에서 서버의 큰 모델을 시뮬레이션. pretrained이므로
    proxy만으로도 Upper에 근접한 강한 Student 구성.
    """
    if pretrained:
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = tvm.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Phase별 역할 → 모델 매핑
def build_model_for_role(role: str, phase: int,
                         num_classes: int = 100, pretrained: bool = True):
    """Phase와 역할(role)에 맞는 모델을 반환.

    Phase 1 (동일 모델): teacher/student/bounds 모두 ResNet-18
    Phase 2 (small→large): teacher=MobileNetV2, student/bounds=ResNet-50
    """
    assert role in ("teacher", "student", "bounds")
    assert phase in (1, 2)

    if phase == 1:
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    else:  # phase == 2
        if role == "teacher":
            return build_mobilenet_v2(num_classes=num_classes, pretrained=pretrained)
        else:
            return build_resnet50(num_classes=num_classes, pretrained=pretrained)


# =============================================================================
# 학습 / 평가 기본 루틴
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes=100):
    """Global accuracy + per-class metrics (precision, recall, f1)."""
    model.eval()
    all_preds = []
    all_labels = []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        logits = model(x)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = float((preds == labels).mean())

    # per-class metrics
    per_class = []
    for c in range(num_classes):
        true_c = (labels == c)
        pred_c = (preds == c)
        tp = int((true_c & pred_c).sum())
        fp = int((~true_c & pred_c).sum())
        fn = int((true_c & ~pred_c).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class.append({
            "class": c, "precision": prec, "recall": rec, "f1": f1,
            "support": int(true_c.sum()),
        })
    macro = {
        "precision": float(np.mean([p["precision"] for p in per_class])),
        "recall": float(np.mean([p["recall"] for p in per_class])),
        "f1": float(np.mean([p["f1"] for p in per_class])),
    }
    return {"accuracy": acc, "per_class": per_class, "macro": macro}


# =============================================================================
# Teacher logit 사전 수집
# =============================================================================
@torch.no_grad()
def collect_logits(model, loader, device):
    """Dataset 순서를 유지한 채 logit을 수집.
    loader는 반드시 shuffle=False여야 함.
    """
    model.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1]
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


# =============================================================================
# Logit 품질 지표 (Layer 2)
# =============================================================================
@torch.no_grad()
def logit_quality_metrics(logits: torch.Tensor, labels: torch.Tensor,
                          expertise_f1: np.ndarray = None,
                          expert_threshold: float = 0.5):
    """Logit 품질 지표 — dark knowledge 유무 확인용.

    Args:
      logits: (N, C) — Teacher 한 명의 proxy logit
      labels: (N,)   — proxy ground truth
      expertise_f1: (C,) — 이 Teacher의 per-class F1 (전문 vs 비전문 구분용)
      expert_threshold: F1 > threshold면 전문 클래스로 분류

    Returns:
      dict with:
        mean_entropy, std_entropy       : 전체 평균 엔트로피
        mean_top1_conf, mean_top2_gap   : softmax 분포 특성
        top1_conf_hist                  : 히스토그램용 bin값들
        expert_entropy / nonexpert_entropy (분리 가능한 경우)
    """
    probs = torch.softmax(logits, dim=1)
    # entropy (nat)
    eps = 1e-12
    entropy = -(probs * (probs + eps).log()).sum(dim=1)
    # top-1 confidence
    top2 = probs.topk(2, dim=1).values
    top1_conf = top2[:, 0]
    top2_gap = top2[:, 0] - top2[:, 1]

    metrics = {
        "mean_entropy": float(entropy.mean()),
        "std_entropy": float(entropy.std()),
        "mean_top1_conf": float(top1_conf.mean()),
        "mean_top2_gap": float(top2_gap.mean()),
        # CDF용 raw values (다운샘플하여 저장)
        "top1_conf_values": top1_conf.numpy()[:5000].tolist(),
        "entropy_values": entropy.numpy()[:5000].tolist(),
    }

    # 전문/비전문 분리 지표
    if expertise_f1 is not None:
        labels_np = labels.numpy()
        expert_classes = np.where(expertise_f1 > expert_threshold)[0]
        nonexpert_classes = np.where(expertise_f1 <= expert_threshold)[0]

        expert_mask = np.isin(labels_np, expert_classes)
        nonexpert_mask = np.isin(labels_np, nonexpert_classes)

        if expert_mask.sum() > 0:
            metrics["expert_mean_entropy"] = float(entropy[expert_mask].mean())
            metrics["expert_mean_top1_conf"] = float(top1_conf[expert_mask].mean())
            metrics["expert_n_samples"] = int(expert_mask.sum())
        if nonexpert_mask.sum() > 0:
            metrics["nonexpert_mean_entropy"] = float(entropy[nonexpert_mask].mean())
            metrics["nonexpert_mean_top1_conf"] = float(top1_conf[nonexpert_mask].mean())
            metrics["nonexpert_n_samples"] = int(nonexpert_mask.sum())

    return metrics


# =============================================================================
# JSON 저장 헬퍼
# =============================================================================
def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
