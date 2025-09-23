# -*- coding: utf-8 -*-
"""
Single-purpose trainer for voice-based HC/PD/MSA/PSP classification (+ APD 그룹핑 지원)

주요 기능
- 파일명 라벨(HC|PD|MSA|PSP 접두어 + *_clean*.wav) 기반 데이터 수집
- 멀티브랜치 모델(CNN(mel) + BiGRU(MFCC+Δ+ΔΔ) + MLP(tabular-lite))
- SpecAugment / 파형 증강 / class-balanced 샘플링 / Label Smoothing / AdamW / warmup+cosine
- feature cache / AMP / Stratified K-Fold
- 라벨 그룹핑(include, rename_map)
- 이진/다중 분류 AUC 계산 및 저장
- 설정 JSON 지원: --config CONFIG.json

Run 예시
  python train_multibranch_voice_pd.py --config configs/hc_vs_msa_strong.json
또는
  python train_multibranch_voice_pd.py --data_root ./DATA/records --out ./runs/mbvpd \
    --epochs 10 --batch 32 --folds 2 --device cuda --amp \
    --aug_spec --spec_p 0.7 --spec_freq_masks 2 --spec_time_masks 2 --spec_F 12 --spec_T_pct 0.1 \
    --aug_wave --wave_p 0.3 --wave_gain_db 3.0 --wave_pitch_semitones 0.5 \
    --sampler weighted --label_smoothing 0.05 --dropout 0.45 --weight_decay 2e-4 \
    --sched cosine --warmup_epochs 1 \
    --include HC,MSA --rename_map_json "{\"MSA\":\"APD\"}"
"""

from __future__ import annotations
import os
import re
import json
import time
import random
import argparse
from pathlib import Path
import sys
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import lr_scheduler as LRS
from tqdm import tqdm


# -------------------------
# Config & Seed
# -------------------------
@dataclass
class Config:
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    fmin: int = 20
    fmax: int = 7600
    n_mfcc: int = 40
    cnn_frames: int = 256
    rnn_frames: int = 300
    seed: int = 42

CFG = Config()
random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# CUDA check & logging
# -------------------------
def check_cuda_and_log(requested_device: str) -> str:
    print("[INFO] Python:", sys.executable)
    print(f"[INFO] torch {torch.__version__} | torch.cuda.is_available()={torch.cuda.is_available()} | torch.version.cuda={torch.version.cuda}")
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        print("[ERROR] '--device cuda' 요청했지만 CUDA 가용하지 않음. GPU 빌드/드라이버를 확인하세요.")
        raise SystemExit(2)
    device = 'cuda' if (requested_device.startswith('cuda') and torch.cuda.is_available()) else 'cpu'
    if device == 'cuda':
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024**3)
        print(f"[INFO] GPU: {name} | VRAM: {total_gb:.1f} GB | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<all>')}")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = torch.randn(2048, 2048, device='cuda')
        for _ in range(10):
            x = x @ x
        torch.cuda.synchronize()
        print(f"[INFO] CUDA matmul sanity: {time.perf_counter()-t0:.3f}s for 10x 2048^2")
        torch.backends.cudnn.benchmark = True
    else:
        print("[WARN] CPU 모드로 실행합니다.")
    return device


# -------------------------
# File listing (filename labels only)
# -------------------------
def list_files_by_name(
    data_dir: str,
    pattern: str = "*_clean*.wav",
    include: Optional[List[str]] = None,
    rename_map: Optional[Dict[str, str]] = None,
    allowed_labels: Tuple[str, ...] = ("HC","PD","MSA","PSP","APD"),
) -> Tuple[List[str], List[str]]:
    """
    - 파일명 접두 대문자 토큰을 라벨로 사용 (예: HC10a1_clean.wav -> 'HC')
    - include: 원본 라벨 기준 필터 (예: ["HC","MSA","PSP"])
    - rename_map: 라벨 리네임 (예: {"MSA":"APD","PSP":"APD"}) → 이진(HC vs APD) 가능
    - allowed_labels: 허용 라벨(기본에 'APD' 추가)
    """
    files = sorted(Path(data_dir).rglob(pattern))
    paths, labels = [], []
    inc = set(s.upper() for s in include) if include else None
    rmap = {k.upper(): v.upper() for k, v in (rename_map or {}).items()}

    for p in files:
        name_u = p.name.upper()
        m = re.match(r"([A-Z]+)", name_u)  # 맨 앞 대문자 토큰
        if not m:
            continue
        lab0 = m.group(1)
        if inc is not None and lab0 not in inc:
            continue
        lab = rmap.get(lab0, lab0)
        if lab not in allowed_labels:
            continue
        paths.append(str(p))
        labels.append(lab)

    if len(paths) == 0:
        raise SystemExit((
            f"[ERROR] 매칭 파일 0개 — data_dir='{data_dir}', pattern='{pattern}'\n"
            " - 예시 파일명: HC1a1_clean.wav / PD3i2_clean.wav\n"
            " - 파일명이 HC|PD|MSA|PSP 로 시작하고 '_clean' 문자열을 포함해야 합니다."
        ))
    return paths, labels


def auto_guess_data_root() -> Optional[str]:
    candidates = [
        "./DATA/records", "./DATA/wav", "./DATA",
        str(Path.cwd()/"DATA"/"records"),
        str(Path.cwd()/"DATA"/"wav"),
        str(Path.cwd()/"DATA"),
    ]
    for d in candidates:
        dpath = Path(d)
        if dpath.is_dir() and any(dpath.rglob("*_clean*.wav")):
            return str(dpath)
    return None


# -------------------------
# Feature cache helpers
# -------------------------
def _hash_for_path(fp: str) -> str:
    h = hashlib.md5()
    h.update(fp.encode('utf-8'))
    key = f"sr={CFG.sample_rate}|mels={CFG.n_mels}|mfcc={CFG.n_mfcc}|nfft={CFG.n_fft}|hop={CFG.hop_length}|win={CFG.win_length}|fmin={CFG.fmin}|fmax={CFG.fmax}"
    h.update(key.encode('utf-8'))
    return h.hexdigest()


# -------------------------
# Features & Augmentations
# -------------------------
def center_crop_or_pad(spec: np.ndarray, target_T: int) -> np.ndarray:
    n_mels, T = spec.shape
    if T == target_T:
        return spec
    if T > target_T:
        start = (T - target_T) // 2
        return spec[:, start:start+target_T]
    pad_total = target_T - T
    left = pad_total // 2
    right = pad_total - left
    return np.pad(spec, ((0,0),(left,right)), mode='constant')


def seq_crop_or_pad(x: np.ndarray, target_T: int) -> np.ndarray:
    T, D = x.shape
    if T == target_T:
        return x
    if T > target_T:
        start = (T - target_T) // 2
        return x[start:start+target_T, :]
    pad_total = target_T - T
    left = pad_total // 2
    right = pad_total - left
    return np.pad(x, ((left,right),(0,0)), mode='edge')


def extract_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = librosa.effects.preemphasis(y)
    yt, _ = librosa.effects.trim(y, top_db=30)
    if len(yt) < sr // 2:
        yt = y

    # mel-spectrogram (CNN branch)
    S = librosa.feature.melspectrogram(
        y=yt, sr=sr, n_fft=CFG.n_fft, hop_length=CFG.hop_length,
        win_length=CFG.win_length, n_mels=CFG.n_mels, fmin=CFG.fmin, fmax=CFG.fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # MFCC(+Δ,+ΔΔ) sequence (RNN branch)
    mfcc = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=CFG.n_mfcc,
                                n_fft=CFG.n_fft, hop_length=CFG.hop_length,
                                fmin=CFG.fmin, fmax=CFG.fmax).T
    mfcc_d  = librosa.feature.delta(mfcc, axis=0)
    mfcc_dd = librosa.feature.delta(mfcc, order=2, axis=0)
    Tm = min(mfcc.shape[0], mfcc_d.shape[0], mfcc_dd.shape[0])
    mfcc_seq = np.concatenate([mfcc[:Tm], mfcc_d[:Tm], mfcc_dd[:Tm]], axis=1)

    # Tabular-lite features (MLP branch)
    f0, _, _ = librosa.pyin(yt, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                            frame_length=CFG.n_fft, hop_length=CFG.hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)
    rms = librosa.feature.rms(y=yt, frame_length=CFG.n_fft, hop_length=CFG.hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=yt, frame_length=CFG.n_fft, hop_length=CFG.hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=yt, sr=sr, n_fft=CFG.n_fft, hop_length=CFG.hop_length)[0]

    def s2(x):
        return np.array([np.mean(x), np.std(x)], dtype=np.float32)

    tab = np.concatenate([s2(f0), s2(rms), s2(zcr), s2(centroid)]).astype(np.float32)

    return S_db.astype(np.float32), mfcc_seq.astype(np.float32), tab


def augment_waveform(y: np.ndarray, sr: int, gain_db: float, pitch_semitones: float) -> np.ndarray:
    if gain_db != 0:
        y = y * (10.0 ** (gain_db / 20.0))
    if abs(pitch_semitones) > 1e-6:
        try:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_semitones)
        except Exception:
            pass
    y = np.clip(y, -1.0, 1.0)
    return y


def spec_augment_mel(x: torch.Tensor, freq_masks: int, F: int, time_masks: int, T_pct: float, p: float) -> torch.Tensor:
    if p <= 0 or (freq_masks <= 0 and time_masks <= 0):
        return x
    B, C, n_mels, T = x.shape
    import random as _rnd
    if _rnd.random() > p:
        return x
    for b in range(B):
        for _ in range(freq_masks):
            f = _rnd.randint(0, max(1, min(F, n_mels//6)))
            f0 = _rnd.randint(0, max(0, n_mels - f))
            x[b, :, f0:f0+f, :] = 0
        for _ in range(time_masks):
            t = _rnd.randint(0, max(1, int(T * T_pct)))
            t0 = _rnd.randint(0, max(0, T - t))
            x[b, :, :, t0:t0+t] = 0
    return x


def time_mask_seq(x: torch.Tensor, time_masks: int, T_pct: float, p: float) -> torch.Tensor:
    if p <= 0 or time_masks <= 0:
        return x
    B, T, D = x.shape
    import random as _rnd
    if _rnd.random() > p:
        return x
    for b in range(B):
        for _ in range(time_masks):
            t = _rnd.randint(0, max(1, int(T * T_pct)))
            t0 = _rnd.randint(0, max(0, T - t))
            x[b, t0:t0+t, :] = 0
    return x


# -------------------------
# Dataset (with cache + optional wave augment)
# -------------------------
class VoiceDataset(Dataset):
    def __init__(self,
                 files: List[str],
                 labels: List[int],
                 sr: int,
                 cache_dir: Optional[str] = None,
                 cache_rebuild: bool = False,
                 use_cache: bool = True,
                 aug_wave: bool = False,
                 wave_p: float = 0.3,
                 wave_gain_db: float = 3.0,
                 wave_pitch_semitones: float = 0.5):
        self.files = files
        self.labels = labels
        self.sr = sr
        self.use_cache = use_cache and (cache_dir is not None)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_rebuild = cache_rebuild
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.aug_wave = aug_wave
        self.wave_p = wave_p
        self.wave_gain_db = wave_gain_db
        self.wave_pitch_semitones = wave_pitch_semitones

    def __len__(self):
        return len(self.files)

    def _cache_path(self, fp: str) -> Path:
        return self.cache_dir / f"{_hash_for_path(fp)}.npz"

    def _load_cached(self, fp: str):
        if not self.use_cache:
            return None
        cpath = self._cache_path(fp)
        if not cpath.exists() or self.cache_rebuild:
            return None
        try:
            z = np.load(cpath)
            return z['mel'], z['mfcc'], z['tab']
        except Exception:
            return None

    def _save_cache(self, fp: str, mel: np.ndarray, mfcc_seq: np.ndarray, tab: np.ndarray):
        if not self.use_cache:
            return
        cpath = self._cache_path(fp)
        try:
            np.savez_compressed(cpath, mel=mel, mfcc=mfcc_seq, tab=tab)
        except Exception:
            pass

    def __getitem__(self, i):
        fp = self.files[i]
        apply_wave = self.aug_wave and (random.random() < self.wave_p)

        if apply_wave:
            y, _ = librosa.load(fp, sr=CFG.sample_rate, mono=True)
            y = augment_waveform(y, self.sr, gain_db=(random.uniform(-self.wave_gain_db, self.wave_gain_db)),
                                 pitch_semitones=(random.uniform(-self.wave_pitch_semitones, self.wave_pitch_semitones)))
            mel, mfcc_seq, tab = extract_features(y, CFG.sample_rate)
        else:
            cached = self._load_cached(fp)
            if cached is not None:
                mel, mfcc_seq, tab = cached
            else:
                y, _ = librosa.load(fp, sr=CFG.sample_rate, mono=True)
                mel, mfcc_seq, tab = extract_features(y, CFG.sample_rate)
                self._save_cache(fp, mel, mfcc_seq, tab)

        mel_norm = (mel + 80.0) / 80.0
        mel_fix = center_crop_or_pad(mel_norm, CFG.cnn_frames)
        mfcc_fix = seq_crop_or_pad(mfcc_seq, CFG.rnn_frames)

        return {
            'mel': torch.from_numpy(mel_fix).unsqueeze(0),    # [1, n_mels, T]
            'mfcc_seq': torch.from_numpy(mfcc_fix),           # [T, D]
            'tab': torch.from_numpy(tab),
            'y': torch.tensor(self.labels[i], dtype=torch.long),
            'path': fp
        }


# -------------------------
# Model
# -------------------------
class CNNBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(128, out_dim)
    def forward(self, x):
        h = self.fe(x).flatten(1)
        return self.proj(h)


class RNNBranch(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128, out_dim: int = 128):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hid, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hid*2, out_dim)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.out(out[:, -1, :])


class MLPBranch(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.45):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.LayerNorm(128), nn.Dropout(dropout),
            nn.Linear(128, out_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)


class MultiBranchNet(nn.Module):
    def __init__(self, tab_dim: int, num_classes: int, head_dropout: float = 0.45):
        super().__init__()
        self.cnn = CNNBranch(128)
        self.rnn = RNNBranch(CFG.n_mfcc*3, 128, 128)
        self.mlp = MLPBranch(tab_dim, 64, dropout=head_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(128+128+64, 256), nn.ReLU(), nn.Dropout(head_dropout),
            nn.Linear(256, num_classes)
        )
    def encode(self, mel, mfcc_seq, tab):
        return torch.cat([self.cnn(mel), self.rnn(mfcc_seq), self.mlp(tab)], dim=1)
    def forward(self, mel, mfcc_seq, tab):
        return self.classifier(self.encode(mel, mfcc_seq, tab))


# -------------------------
# Train / Eval
# -------------------------
def _worker_init(worker_id: int):
    seed = CFG.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_tqdm: bool = True,
                    amp: bool = False,
                    spec_cfg: Optional[dict] = None,
                    seqaug_cfg: Optional[dict] = None,
                    clip_grad: Optional[float] = None):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []
    iterable = tqdm(loader, total=len(loader), desc='[Train]', dynamic_ncols=True) if use_tqdm else loader
    first_batch = True
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for batch in iterable:
        mel = batch['mel'].to(device, non_blocking=True)
        mfcc = batch['mfcc_seq'].to(device, non_blocking=True)
        tab = batch['tab'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        if first_batch:
            print("[INFO] tensor devices:", mel.device, mfcc.device, tab.device)
            first_batch = False

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            if spec_cfg is not None:
                mel = spec_augment_mel(mel,
                                       spec_cfg['freq_masks'], spec_cfg['F'],
                                       spec_cfg['time_masks'], spec_cfg['T_pct'], spec_cfg['p'])
            if seqaug_cfg is not None:
                mfcc = time_mask_seq(mfcc, seqaug_cfg['time_masks'], seqaug_cfg['T_pct'], seqaug_cfg['p'])

            logits = model(mel, mfcc, tab)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        if clip_grad is not None and clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach().item() * y.size(0)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(logits.detach().argmax(1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return total_loss/len(loader.dataset), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_tqdm: bool = True):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
    iterable = tqdm(loader, total=len(loader), desc='[Val]', dynamic_ncols=True, leave=False) if use_tqdm else loader
    for batch in iterable:
        mel = batch['mel'].to(device, non_blocking=True)
        mfcc = batch['mfcc_seq'].to(device, non_blocking=True)
        tab = batch['tab'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        logits = model(mel, mfcc, tab)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        prob = F.softmax(logits, 1)
        y_prob.append(prob.cpu().numpy())
        y_true.append(y.cpu().numpy())
        y_pred.append(prob.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    return total_loss/len(loader.dataset), accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro'), y_true, y_pred, y_prob


# -------------------------
# Main training (filename labels only)
# -------------------------
def run_training(data_root: str,
                 out_dir: str,
                 epochs: int = 30,
                 batch_size: int = 16,
                 folds: int = 5,
                 lr: float = 3e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 pattern: str = "*_clean*.wav",
                 use_tqdm: bool = True,
                 amp: bool = False,
                 # cache
                 cache_dir: Optional[str] = "./feature_cache",
                 cache_rebuild: bool = False,
                 no_cache: bool = False,
                 # augment: SpecAugment (mel) + seq mask
                 aug_spec: bool = True,
                 spec_p: float = 0.7,
                 spec_freq_masks: int = 2,
                 spec_time_masks: int = 2,
                 spec_F: int = 12,
                 spec_T_pct: float = 0.10,
                 aug_seq: bool = True,
                 seq_p: float = 0.7,
                 seq_time_masks: int = 1,
                 seq_T_pct: float = 0.05,
                 # augment: waveform (gain/pitch)
                 aug_wave: bool = True,
                 wave_p: float = 0.3,
                 wave_gain_db: float = 3.0,
                 wave_pitch_semitones: float = 0.5,
                 # optimization
                 dropout: float = 0.45,
                 label_smoothing: float = 0.05,
                 weight_decay: float = 2e-4,
                 clip_grad: Optional[float] = None,
                 # sampler
                 sampler_type: str = 'weighted',  # 'none' | 'weighted'
                 # scheduler
                 sched: str = 'cosine',  # 'cosine' | 'plateau'
                 warmup_epochs: int = 3,
                 # 라벨 필터/그룹핑
                 include: Optional[List[str]] = None,
                 rename_map: Optional[Dict[str, str]] = None,
                 ):

    os.makedirs(out_dir, exist_ok=True)

    files, y_labels = list_files_by_name(data_root, pattern, include=include, rename_map=rename_map)
    classes = sorted(list({lab for lab in y_labels}))
    label2idx = {lab:i for i, lab in enumerate(classes)}
    y = np.array([label2idx[lab] for lab in y_labels], dtype=np.int64)

    # pre-scan to get tab_dim
    y0, _ = librosa.load(files[0], sr=CFG.sample_rate, mono=True)
    _, _, tab0 = extract_features(y0, CFG.sample_rate)
    tab_dim = tab0.shape[0]

    print(f"[CACHE] {'enabled' if not no_cache and cache_dir else 'disabled'} | dir={cache_dir} | rebuild={cache_rebuild}")
    print(f"[AUG] spec={aug_spec} p={spec_p} | wave={aug_wave} p={wave_p} | seq={aug_seq} p={seq_p}")
    print(f"[SCHED] {sched} | warmup_epochs={warmup_epochs}")
    print(f"[OPT] label_smoothing={label_smoothing} dropout={dropout} wd={weight_decay} sampler={sampler_type}")
    print(f"[INFO] classes={classes}")

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=CFG.seed)
    reports = []
    # 전체 OOF 저장용
    oof_true_all, oof_prob_all, oof_pred_all = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(files, y), 1):
        tr_files = [files[i] for i in tr_idx]
        va_files = [files[i] for i in va_idx]
        tr_y = [int(y[i]) for i in tr_idx]
        va_y = [int(y[i]) for i in va_idx]

        ds_tr = VoiceDataset(
            tr_files, tr_y, CFG.sample_rate,
            cache_dir=cache_dir, cache_rebuild=cache_rebuild,
            use_cache=(not no_cache),
            aug_wave=aug_wave, wave_p=wave_p,
            wave_gain_db=wave_gain_db, wave_pitch_semitones=wave_pitch_semitones,
        )
        ds_va = VoiceDataset(
            va_files, va_y, CFG.sample_rate,
            cache_dir=cache_dir, cache_rebuild=False,
            use_cache=(not no_cache),
            aug_wave=False,  # 검증에는 파형 증강 금지
        )

        # Sampler
        if sampler_type == 'weighted':
            class_counts = np.bincount(tr_y, minlength=len(classes))
            inv = 1.0/np.clip(class_counts, 1, None)
            sample_weights = np.array([inv[c] for c in tr_y], dtype=np.float64)
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler,
                               num_workers=min(12, os.cpu_count() or 4), pin_memory=device.startswith('cuda'),
                               persistent_workers=True, prefetch_factor=4, worker_init_fn=_worker_init)
        else:
            dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                               num_workers=min(12, os.cpu_count() or 4), pin_memory=device.startswith('cuda'),
                               persistent_workers=True, prefetch_factor=4, worker_init_fn=_worker_init)

        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                           num_workers=min(12, os.cpu_count() or 4), pin_memory=device.startswith('cuda'),
                           persistent_workers=True, prefetch_factor=4, worker_init_fn=_worker_init)

        model = MultiBranchNet(tab_dim=tab_dim, num_classes=len(classes), head_dropout=dropout).to(device)
        print("[INFO] model.device:", next(model.parameters()).device)

        # Loss (class weights + label smoothing)
        class_counts = np.bincount(tr_y, minlength=len(classes))
        inv = 1.0/np.clip(class_counts, 1, None)
        weights = inv/inv.sum()*len(classes)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                        label_smoothing=label_smoothing)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if sched == 'cosine':
            warmup = max(0, min(warmup_epochs, epochs))
            sched1 = LRS.LinearLR(optimizer, start_factor=0.1, total_iters=warmup) if warmup > 0 else LRS.ConstantLR(optimizer, factor=1.0, total_iters=1)
            sched2 = LRS.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
            scheduler = LRS.SequentialLR(optimizer, schedulers=[sched1, sched2], milestones=[warmup])
            sched_on_metric = False
        else:
            scheduler = LRS.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
            sched_on_metric = True

        best_f1 = -1.0
        patience = 8
        ckpt_path = os.path.join(out_dir, f"model_fold{fold}.pt")
        hist = []

        spec_cfg = {'freq_masks': spec_freq_masks, 'F': spec_F, 'time_masks': spec_time_masks, 'T_pct': spec_T_pct, 'p': spec_p} if aug_spec else None
        seqaug_cfg = {'time_masks': seq_time_masks, 'T_pct': seq_T_pct, 'p': seq_p} if aug_seq else None

        for ep in range(1, epochs+1):
            tl, ta, tf = train_one_epoch(model, dl_tr, criterion, optimizer, device,
                                         use_tqdm=use_tqdm, amp=amp,
                                         spec_cfg=spec_cfg, seqaug_cfg=seqaug_cfg,
                                         clip_grad=clip_grad)
            vl, va, vf, y_true, y_pred, y_prob = evaluate(model, dl_va, criterion, device, use_tqdm=use_tqdm)

            if sched_on_metric:
                scheduler.step(vf)
            else:
                scheduler.step()

            hist.append({'epoch': ep, 'tr_loss': tl, 'tr_acc': ta, 'tr_f1': tf, 'va_loss': vl, 'va_acc': va, 'va_f1': vf,
                         'lr': optimizer.param_groups[0]['lr']})
            print(f"Fold {fold} | Epoch {ep:03d} | TR acc {ta:.3f} f1 {tf:.3f} | VA acc {va:.3f} f1 {vf:.3f}")

            if vf > best_f1:
                best_f1 = vf
                torch.save({'model': model.state_dict(), 'label2idx': label2idx, 'cfg': CFG.__dict__, 'tab_dim': tab_dim}, ckpt_path)
                patience = 8
            else:
                patience -= 1
                if patience <= 0:
                    print("Early stopping")
                    break

        # reload best and final val report
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        vl, va, vf, y_true, y_pred, y_prob = evaluate(model, dl_va, criterion, device)

        # 저장물(리포트/행렬/스코어)
        rpt = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # AUC 계산: 이진이면 bin AUC, 다중이면 OVR macro
        auc_bin = None
        try:
            if len(classes) == 2:
                pos_idx = 1  # 정렬된 classes 기준 두 번째를 양성으로 사용 (예: ['HC','MSA'] → MSA)
                # fold에 단일 클래스만 있을 때 roc_auc_score가 실패할 수 있음 → 예외 처리
                if len(np.unique(y_true)) == 2:
                    auc_bin = roc_auc_score(y_true, y_prob[:, pos_idx])
                    auc_macro = float(auc_bin)
                else:
                    raise ValueError("Validation fold has a single class; AUC undefined.")
            else:
                auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"[WARN] AUC 계산 실패 (fold={fold}): {e}")
            auc_macro = float('nan')

        # fold별 결과 저장
        with open(os.path.join(out_dir, f"report_fold{fold}.txt"), 'w', encoding='utf-8') as f:
            f.write(rpt + "\n")
            f.write(str(cm) + "\n")
            if auc_bin is not None:
                f.write(f"AUC(binary, pos='{classes[1]}') = {auc_bin:.6f}\n")
            f.write(f"AUC(macro/ovr) = {auc_macro:.6f}\n")
            f.write(json.dumps(hist))

        np.save(os.path.join(out_dir, f"fold{fold}_y_true.npy"), y_true)
        np.save(os.path.join(out_dir, f"fold{fold}_y_pred.npy"), y_pred)
        np.save(os.path.join(out_dir, f"fold{fold}_y_prob.npy"), y_prob)

        # 리포트 집계
        rec = {'fold': fold, 'val_acc': float(va), 'val_f1': float(vf), 'val_auc_macro_ovr': float(auc_macro)}
        if auc_bin is not None:
            rec['val_auc_binary'] = float(auc_bin)
        reports.append(rec)

        # OOF 누적
        oof_true_all.append(y_true)
        oof_pred_all.append(y_pred)
        oof_prob_all.append(y_prob)

    # 전체 OOF 저장
    try:
        oof_true_all = np.concatenate(oof_true_all)
        oof_pred_all = np.concatenate(oof_pred_all)
        oof_prob_all = np.concatenate(oof_prob_all, axis=0)
        np.savez_compressed(os.path.join(out_dir, "oof_all.npz"),
                            y_true=oof_true_all, y_pred=oof_pred_all, y_prob=oof_prob_all, classes=np.array(classes))
    except Exception as e:
        print(f"[WARN] OOF 저장 실패: {e}")

    # 요약 저장
    with open(os.path.join(out_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

    # CSV 요약
    try:
        import csv
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            hdr = ["fold", "val_acc", "val_f1", "val_auc_macro_ovr", "val_auc_binary"]
            w.writerow(hdr)
            for r in reports:
                w.writerow([r.get("fold"),
                            r.get("val_acc"),
                            r.get("val_f1"),
                            r.get("val_auc_macro_ovr"),
                            r.get("val_auc_binary","")])
    except Exception as e:
        print(f"[WARN] summary.csv 저장 실패: {e}")

    return reports


# -------------------------
# Main (F5-friendly) + JSON config
# -------------------------
def _coerce_include(v):
    if v is None:
        return None
    if isinstance(v, str):
        return [s.strip().upper() for s in v.split(",") if s.strip()]
    if isinstance(v, list):
        return [str(s).strip().upper() for s in v]
    return None

def _coerce_rename_map(v):
    if v is None:
        return None
    if isinstance(v, str):
        # CLI 호환: --rename_map_json '{"MSA":"APD","PSP":"APD"}'
        try:
            obj = json.loads(v)
        except Exception:
            return None
        return {k.upper(): str(val).upper() for k, val in obj.items()}
    if isinstance(v, dict):
        return {str(k).upper(): str(val).upper() for k, val in v.items()}
    return None


if __name__ == "__main__":
    # 1) 1차 파서: --config만 선 파싱
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None, help='JSON 설정 파일 경로')
    known, _ = pre.parse_known_args()

    # 2) 본 파서
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument('--data_root', type=str, default=None)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--pattern', type=str, default='*_clean*.wav')
    ap.add_argument('--no_tqdm', action='store_true', help='progress bar 끄기')
    ap.add_argument('--amp', action='store_true', help='CUDA AMP 활성화')

    # cache
    ap.add_argument('--cache_dir', type=str, default='./feature_cache')
    ap.add_argument('--cache_rebuild', action='store_true')
    ap.add_argument('--no_cache', action='store_true')

    # augment: SpecAugment + sequence
    ap.add_argument('--aug_spec', action='store_true')
    ap.add_argument('--spec_p', type=float, default=0.7)
    ap.add_argument('--spec_freq_masks', type=int, default=2)
    ap.add_argument('--spec_time_masks', type=int, default=2)
    ap.add_argument('--spec_F', type=int, default=12)
    ap.add_argument('--spec_T_pct', type=float, default=0.10)

    ap.add_argument('--aug_seq', action='store_true')
    ap.add_argument('--seq_p', type=float, default=0.7)
    ap.add_argument('--seq_time_masks', type=int, default=1)
    ap.add_argument('--seq_T_pct', type=float, default=0.05)

    # augment: waveform
    ap.add_argument('--aug_wave', action='store_true')
    ap.add_argument('--wave_p', type=float, default=0.3)
    ap.add_argument('--wave_gain_db', type=float, default=3.0)
    ap.add_argument('--wave_pitch_semitones', type=float, default=0.5)

    # optimization
    ap.add_argument('--dropout', type=float, default=0.45)
    ap.add_argument('--label_smoothing', type=float, default=0.05)
    ap.add_argument('--weight_decay', type=float, default=2e-4)
    ap.add_argument('--clip_grad', type=float, default=0.0)

    # sampler & scheduler
    ap.add_argument('--sampler', type=str, default='weighted', choices=['none','weighted'])
    ap.add_argument('--sched', type=str, default='cosine', choices=['cosine','plateau'])
    ap.add_argument('--warmup_epochs', type=int, default=3)

    # 시나리오 필터/그룹핑
    ap.add_argument('--include', type=str, default=None, help='예: "HC,MSA,PSP" 또는 JSON에서는 ["HC","MSA"]')
    ap.add_argument('--rename_map_json', type=str, default=None, help='CLI 전용: "{\"MSA\":\"APD\",\"PSP\":\"APD\"}"')

    args = ap.parse_args()

    # 3) JSON 설정 로드 → 기본값 덮어쓰기 (단, CLI에서 직접 지정된 값이 더 우선)
    if known.config:
        with open(known.config, "r", encoding="utf-8") as f:
            cfgj = json.load(f)
        # JSON에 들어온 키를 args에 반영 (이미 지정된 CLI 플래그는 유지)
        for k, v in cfgj.items():
            if not hasattr(args, k):
                continue
            cur = getattr(args, k)
            # bool 플래그(True/False 토글)인 경우, JSON값이 있으면 그대로 반영 (CLI의 action='store_true'는 기본 False)
            if isinstance(cur, bool):
                setattr(args, k, bool(v))
            else:
                # CLI가 기본값 그대로이고 JSON에 값이 있으면 교체
                setattr(args, k, v if cur is None or cur == ap.get_default(k) else cur)

    # include / rename_map 파싱 (문자열/리스트/딕셔너리 모두 허용)
    include = _coerce_include(args.include if args.include is not None else getattr(args, "include", None))
    # JSON에 rename_map이 딕셔너리로 올 수 있으므로 함께 처리
    rename_map = None
    if getattr(args, "rename_map", None) is not None:
        rename_map = _coerce_rename_map(args.rename_map)
    elif args.rename_map_json:
        rename_map = _coerce_rename_map(args.rename_map_json)

    # F5 무인자 실행: 기본값 자동 채우기
    if not args.data_root:
        guessed = auto_guess_data_root()
        if guessed:
            args.data_root = guessed
            print(f"[INFO] data_root 자동 감지: {args.data_root}")
        else:
            print("[ERROR] data_root를 찾지 못했습니다. --data_root 인자 또는 ./DATA/records|wav 구조를 준비하세요.")
            raise SystemExit(1)
    if not args.out:
        args.out = "./runs/mbvpd"
        print(f"[INFO] out 기본 설정: {args.out}")

    set_seed(CFG.seed)
    args.device = check_cuda_and_log(args.device)

    run_training(
        data_root=args.data_root,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch,
        folds=args.folds,
        lr=args.lr,
        device=args.device,
        pattern=args.pattern,
        use_tqdm=(not args.no_tqdm),
        amp=args.amp and args.device.startswith('cuda'),
        # cache
        cache_dir=args.cache_dir,
        cache_rebuild=args.cache_rebuild,
        no_cache=args.no_cache,
        # aug
        aug_spec=args.aug_spec,
        spec_p=args.spec_p,
        spec_freq_masks=args.spec_freq_masks,
        spec_time_masks=args.spec_time_masks,
        spec_F=args.spec_F,
        spec_T_pct=args.spec_T_pct,
        aug_seq=args.aug_seq,
        seq_p=args.seq_p,
        seq_time_masks=args.seq_time_masks,
        seq_T_pct=args.seq_T_pct,
        aug_wave=args.aug_wave,
        wave_p=args.wave_p,
        wave_gain_db=args.wave_gain_db,
        wave_pitch_semitones=args.wave_pitch_semitones,
        # optim
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        clip_grad=(args.clip_grad if args.clip_grad and args.clip_grad>0 else None),
        # sampler & sched
        sampler_type=args.sampler,
        sched=args.sched,
        warmup_epochs=args.warmup_epochs,
        # 라벨 필터/그룹핑
        include=include,
        rename_map=rename_map,
    )
