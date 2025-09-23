#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_similar_binary_classification(HC,PSP).py
— Enhanced Features + SVM focus + NaN/Inf guards + Soft-vote Ensemble (optional)
— Constrained threshold tuning (min PSP recall) + CV calibration

기능 요약
- 이진 분류(HC vs PSP), 파일명 프리픽스(HC|PSP), '*_clean*.wav'만 사용
- 특징(강화): MFCC(+Δ +ΔΔ), 스펙트럼 통계, ZCR, F0/jitter(근사), RMS/shimmer(근사),
             HNR(안전), CPP(안전), Formants F1–F3(LPC, 안전), duration/skew/kurtosis
- 증강: 클래스별 배수(--aug_hc/--aug_psp), 병렬, 캐싱, 상한(--max_train_samples)
- 모델: SVM_Linear, SVM_RBF 집중 (RBF 그리드/가중치 확장)
- 보정: none/sigmoid/isotonic (기본: sigmoid), CalibratedClassifierCV(cv=5) 교차보정
- 임계값: metric(accuracy/balanced_accuracy/psp_recall/f1_macro) 최적화 +
          옵션 --min_psp_recall 로 PSP 민감도 하한 제약
- 앙상블(옵션): soft-vote(가중 평균) + 검증셋 기반 가중치/임계값 동시 탐색
- 저장: 모델별 .joblib + 결과 요약 CSV + (옵션) 앙상블 .joblib

Python 3.10.9
필요: numpy, scipy, scikit-learn, librosa, joblib, tqdm
"""
from __future__ import annotations

import re
import csv
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from collections import Counter

import numpy as np
import joblib
import librosa
from scipy.stats import skew, kurtosis
from scipy.signal import lfilter

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, log_loss, accuracy_score,
    balanced_accuracy_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from joblib import Parallel, delayed
from tqdm import tqdm
import random
from numpy.linalg import LinAlgError

# ===== 재현성 =====
try:
    import torch
except Exception:
    torch = None

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
set_seed(42)

# ===== 라벨 =====
CLASSES = ["HC", "PSP"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ===== 유틸: NaN/Inf 가드 =====
def _sanitize_array(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def _is_finite_vec(x: np.ndarray) -> bool:
    return np.all(np.isfinite(x))

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===== 파일 스캔 =====
def list_files_by_name(data_dir: str, pattern: str = "*_clean*.wav") -> Tuple[List[str], List[str]]:
    files = sorted(Path(data_dir).rglob(pattern))
    paths, labels = [], []
    for p in files:
        m = re.match(r"(HC|PSP)", p.name.upper())
        if not m: continue
        paths.append(str(p)); labels.append(m.group(1))
    if not paths:
        raise SystemExit("[ERROR] 매칭 파일 0개 — 파일명이 HC|PSP로 시작하고 '_clean' 포함 필요")
    return paths, labels

def subject_id_from_name(fname: str) -> str:
    m = re.match(r"(HC|PSP)(\d+)", fname.upper())
    return m.group(0) if m else fname.split("_")[0]

# ===== 증강 =====
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))

def add_noise_snr(x: np.ndarray, snr_db_low=15.0, snr_db_high=30.0) -> np.ndarray:
    snr_db = np.random.uniform(snr_db_low, snr_db_high)
    r = _rms(x); snr = 10 ** (snr_db / 20.0)
    noise_rms = r / max(snr, 1e-6)
    y = x + np.random.randn(len(x)).astype(np.float32) * noise_rms
    m = np.max(np.abs(y)) + 1e-12
    return (y/m).astype(np.float32) if m > 1.0 else y.astype(np.float32)

def random_time_shift(x: np.ndarray, sr: int, max_shift_sec: float = 0.15) -> np.ndarray:
    max_shift = int(sr * max_shift_sec)
    if max_shift < 1: return x
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift).astype(np.float32)

def random_gain(x: np.ndarray, db_range=(-3.0, 3.0)) -> np.ndarray:
    g = 10 ** (np.random.uniform(*db_range) / 20.0)
    y = (x * g).astype(np.float32)
    m = np.max(np.abs(y)) + 1e-12
    return (y/m).astype(np.float32) if m > 1.0 else y

def maybe_time_stretch(x: np.ndarray, rate_range=(0.97, 1.03), p=0.3) -> np.ndarray:
    if np.random.rand() > p: return x
    y = librosa.effects.time_stretch(x, rate=float(np.random.uniform(*rate_range)))
    return np.pad(y, (0, len(x)-len(y))) if len(y) < len(x) else y[:len(x)]

def augment_once(y: np.ndarray, sr: int) -> np.ndarray:
    z = y.copy().astype(np.float32)
    if np.random.rand() < 0.7: z = random_time_shift(z, sr, 0.15)
    if np.random.rand() < 0.5: z = random_gain(z, (-3.0, 3.0))
    if np.random.rand() < 0.4: z = add_noise_snr(z, 15.0, 30.0)
    if np.random.rand() < 0.3: z = maybe_time_stretch(z, (0.97, 1.03), p=1.0)
    return z

# ===== 오디오 로드 =====
@dataclass
class FeatConfig:
    sr: int = 16000
    top_db: int = 30
    frame_length: int = 1024
    hop_length: int = 256
    n_mfcc: int = 13
    use_pca: bool = True

def load_audio(path, sr, top_db):
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    if sr is not None and orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    return y.astype(np.float32) if y.size else np.zeros(sr, dtype=np.float32)

# ===== 저수준 피처 =====
def robust_f0(y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> np.ndarray:
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr, frame_length=frame_length, hop_length=hop_length)
    except Exception:
        return np.array([0.0], dtype=np.float32)
    f0 = f0[np.isfinite(f0)]; f0 = f0[f0 > 0]
    return f0.astype(np.float32) if f0.size else np.array([0.0], dtype=np.float32)

def frame_rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True).flatten()
    rms = rms[np.isfinite(rms)]
    return rms.astype(np.float32) if rms.size else np.array([0.0], dtype=np.float32)

# ===== HNR (안전 계산) =====
def hnr_features(y: np.ndarray, sr: int, frame_length: int, hop_length: int):
    try:
        y = _sanitize_array(y)
        y_harm = librosa.effects.harmonic(y)
        y_res  = y - y_harm

        # 글로벌 HNR(dB)
        hnr_db = 10.0 * np.log10((_rms(y_harm) ** 2) / ((_rms(y_res) ** 2) + 1e-12) + 1e-12)

        rms_h = frame_rms(y_harm, frame_length, hop_length)
        rms_r = frame_rms(y_res,  frame_length, hop_length)
        denom = (rms_r ** 2) + 1e-12
        num   = (rms_h ** 2)
        hnr_frames = 10.0 * np.log10(num / denom + 1e-12)
        if hnr_frames.size == 0 or not _is_finite_vec(hnr_frames):
            return np.array([hnr_db, 0.0, 0.0, 0.0], dtype=np.float32)

        out = np.array([
            float(hnr_db),
            float(np.mean(hnr_frames)),
            float(np.std(hnr_frames)),
            float(np.percentile(hnr_frames, 75) - np.percentile(hnr_frames, 25)),
        ], dtype=np.float32)
        return _sanitize_array(out)
    except Exception:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# ===== CPP (안전 계산) =====
def cpp_features(y: np.ndarray, sr: int, frame_length: int, hop_length: int):
    try:
        y = _sanitize_array(y)
        win = np.hanning(frame_length).astype(np.float32)
        qmin = int(round(sr / 400.0))  # ~2.5ms
        qmax = int(round(sr / 50.0))   # ~20ms

        vals = []
        for start in range(0, len(y) - frame_length + 1, hop_length):
            fr = y[start:start + frame_length]
            if np.mean(fr * fr) < 1e-8:
                continue
            fr = fr * win
            spec = np.fft.rfft(fr)
            mag  = np.abs(spec)
            if mag.size == 0:
                continue
            cep = np.fft.irfft(np.log(np.maximum(mag, 1e-8)))
            if not _is_finite_vec(cep) or qmax >= len(cep) or qmin >= qmax:
                continue
            seg = cep[qmin:qmax]
            if seg.size == 0 or not np.all(np.isfinite(seg)):
                continue
            peak_idx = int(np.argmax(seg)) + qmin
            base = np.interp(peak_idx, [qmin, qmax-1], [cep[qmin], cep[qmax-1]])
            cpp_val = float(cep[peak_idx] - base)
            if np.isfinite(cpp_val):
                vals.append(cpp_val)

        if len(vals) == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        vals = np.asarray(vals, dtype=np.float32)
        out = np.array([
            float(np.mean(vals)),
            float(np.std(vals)),
            float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        ], dtype=np.float32)
        return _sanitize_array(out)
    except Exception:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

# ===== Formants via LPC (안전 계산) =====
def formant_features(y: np.ndarray, sr: int, frame_length: int, hop_length: int,
                     n_formants=3, lpc_order=16):
    try:
        y = _sanitize_array(y)
        y_pe = lfilter([1.0, -0.97], [1.0], y).astype(np.float32)
        win = np.hanning(frame_length).astype(np.float32)

        F_lists = [[] for _ in range(n_formants)]
        for start in range(0, len(y_pe) - frame_length + 1, hop_length):
            fr = y_pe[start:start + frame_length]
            if np.mean(fr * fr) < 1e-7:
                continue
            fr = fr * win
            if not _is_finite_vec(fr):
                continue

            ord_use = min(lpc_order, max(8, len(fr)//2 - 1))
            try:
                a = librosa.lpc(fr, order=ord_use)
            except (LinAlgError, ValueError, FloatingPointError):
                continue
            if a is None or len(a) == 0 or not _is_finite_vec(a) or abs(a[0]) < 1e-8:
                continue

            rts = np.roots(a)
            if rts.size == 0:
                continue
            rts = rts[np.imag(rts) >= 0.0]
            if rts.size == 0:
                continue
            angs = np.angle(rts)
            freqs = angs * (sr / (2.0 * np.pi))
            freqs = freqs[(freqs > 90.0) & (freqs < 5000.0)]
            freqs = np.sort(np.real(freqs))
            if freqs.size >= n_formants:
                for i in range(n_formants):
                    Fi = float(freqs[i])
                    if np.isfinite(Fi):
                        F_lists[i].append(Fi)

        feats = []
        for Fi in F_lists:
            if len(Fi) == 0:
                feats.extend([0.0, 0.0])
            else:
                arr = np.array(Fi, dtype=np.float32)
                feats.extend([float(np.mean(arr)), float(np.std(arr))])

        return _sanitize_array(np.array(feats, dtype=np.float32))
    except Exception:
        return np.zeros(n_formants * 2, dtype=np.float32)

# ===== 메인 피처 추출 =====
def extract_features(y: np.ndarray, sr: int, cfg: FeatConfig) -> np.ndarray:
    # MFCC + Δ + ΔΔ
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc)
    d1   = librosa.feature.delta(mfcc, order=1)
    d2   = librosa.feature.delta(mfcc, order=2)
    feats = []
    for mat in (mfcc, d1, d2):
        feats.append(np.mean(mat, axis=1))
        feats.append(np.std(mat, axis=1))
    feats = np.concatenate(feats, axis=0)

    # Spectrum stats
    cent = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr).flatten()
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).flatten()
    flat = librosa.feature.spectral_flatness(y=y).flatten()
    spec_stats = []
    for vec in (cent, bw, roll, flat):
        spec_stats += [np.mean(vec), np.std(vec)]
    feats = np.concatenate([feats, np.array(spec_stats, dtype=np.float32)])

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y=y).flatten()
    feats = np.concatenate([feats, [np.mean(zcr), np.std(zcr)]])

    # F0 + jitter(근사)
    f0 = robust_f0(y, sr, cfg.frame_length, cfg.hop_length)
    if f0.size < 3:
        f0_stats = [np.mean(f0), 0.0, 0.0, 0.0]
    else:
        df0 = np.diff(f0)
        jitter = (np.std(df0) / (np.mean(f0) + 1e-8)) * 100.0
        f0_stats = [float(np.mean(f0)), float(np.std(f0)),
                    float(np.percentile(f0, 75) - np.percentile(f0, 25)), float(jitter)]
    feats = np.concatenate([feats, np.array(f0_stats, dtype=np.float32)])

    # RMS + shimmer(근사)
    rms = frame_rms(y, cfg.frame_length, cfg.hop_length)
    if rms.size < 3:
        rms_stats = [np.mean(rms), 0.0, 0.0, 0.0]
    else:
        drms = np.diff(rms)
        shimmer = (np.std(drms) / (np.mean(rms) + 1e-8)) * 100.0
        rms_stats = [float(np.mean(rms)), float(np.std(rms)),
                     float(np.percentile(rms, 75) - np.percentile(rms, 25)), float(shimmer)]
    feats = np.concatenate([feats, np.array(rms_stats, dtype=np.float32)])

    # Enhanced: HNR / CPP / Formants
    feats = np.concatenate([feats, hnr_features(y, sr, cfg.frame_length, cfg.hop_length)])        # 4
    feats = np.concatenate([feats, cpp_features(y, sr, cfg.frame_length, cfg.hop_length)])        # 3
    feats = np.concatenate([feats, formant_features(y, sr, cfg.frame_length, cfg.hop_length,
                                                    n_formants=3, lpc_order=16)])                # 6

    # Duration + shape stats (안전 변환)
    dur = len(y) / sr
    s = float(np.nan_to_num(skew(y), nan=0.0, posinf=0.0, neginf=0.0))
    k = float(np.nan_to_num(kurtosis(y), nan=0.0, posinf=0.0, neginf=0.0))
    feats = np.concatenate([feats, np.array([dur, s, k], dtype=np.float32)])

    # 전체 벡터 sanitize
    feats = _sanitize_array(feats)
    return feats

# ===== 병렬 증강/특징 =====
def _extract_features_from_wave(y_sig: np.ndarray, sr: int, cfg: FeatConfig) -> np.ndarray:
    return extract_features(y_sig, sr, cfg)

def _augment_and_extract_many(y_sig: np.ndarray, sr: int, cfg: FeatConfig,
                              n_aug: int, subj_id: str, lab_idx: int):
    out_X, out_y, out_sub = [], [], []
    out_X.append(_extract_features_from_wave(y_sig, sr, cfg)); out_y.append(lab_idx); out_sub.append(subj_id)
    for _ in range(max(0, n_aug)):
        y_aug = augment_once(y_sig, sr)
        out_X.append(_extract_features_from_wave(y_aug, sr, cfg)); out_y.append(lab_idx); out_sub.append(subj_id)
    return np.stack(out_X, axis=0), np.array(out_y, dtype=int), np.array(out_sub)

def build_feature_matrix_aug_per_class_parallel(
    paths: List[str], labels: List[str], cfg: FeatConfig,
    aug_hc: int = 0, aug_psp: int = 0, n_jobs: int = -1,
    preload_audio: bool = True, max_train_samples: int | None = None
):
    waves = {}
    if preload_audio:
        for p in tqdm(paths, desc="[Cache] preload train waves", ncols=100):
            waves[p] = load_audio(p, cfg.sr, cfg.top_db)

    def runner(p, lab):
        try:
            y_sig = waves[p] if preload_audio else load_audio(p, cfg.sr, cfg.top_db)
            subj = subject_id_from_name(Path(p).name)
            lab_idx = CLASS_TO_IDX[lab]
            A = aug_psp if lab == "PSP" else aug_hc
            return _augment_and_extract_many(y_sig, cfg.sr, cfg, A, subj, lab_idx)
        except Exception as e:
            print(f"[WARN] feature extraction failed for {p}: {e}")
            return None

    results = Parallel(n_jobs=n_jobs if n_jobs != 0 else 1, backend="loky", batch_size=1)(
        delayed(runner)(p, lab) for p, lab in tqdm(list(zip(paths, labels)), total=len(paths),
                                                   desc=f"[Extract+Aug/Parallel] HC={aug_hc}, PSP={aug_psp}", ncols=100)
    )
    results = [r for r in results if r is not None]
    if len(results) == 0:
        raise SystemExit("[ERROR] 모든 샘플의 특징 추출에 실패했습니다.")

    X = np.concatenate([r[0] for r in results], axis=0)
    y = np.concatenate([r[1] for r in results], axis=0)
    subs = np.concatenate([r[2] for r in results], axis=0)

    if (max_train_samples is not None) and (X.shape[0] > max_train_samples):
        keep_idx = []
        rng = np.random.default_rng(42)
        for c in np.unique(y):
            idx_c = np.where(y == c)[0]
            n_keep_c = int(round(max_train_samples * (len(idx_c) / len(y))))
            keep_idx.append(rng.choice(idx_c, size=max(1, min(n_keep_c, len(idx_c))), replace=False))
        keep_idx = np.concatenate(keep_idx)
        X, y, subs = X[keep_idx], y[keep_idx], subs[keep_idx]
    return X, y, subs

# ===== 임계값 튜닝 (제약식 지원) =====
def tune_threshold(y_true: np.ndarray, proba_psp: np.ndarray, metric: str,
                   min_psp_recall: float | None = None) -> float:
    """
    metric 기준으로 최적 임계값 선택.
    min_psp_recall 지정 시, Recall_psp >= min_psp_recall 제약을 만족하는 후보 중 metric 최대.
    후보 없으면 PSP recall 최대치의 임계값으로 fallback.
    """
    ths = np.linspace(0.05, 0.95, 181)
    cand = []
    best_t, best_s = 0.5, -1.0

    for t in ths:
        y_pred = (proba_psp >= t).astype(int)  # PSP=1
        recP = recall_score(y_true, y_pred, pos_label=CLASS_TO_IDX["PSP"])
        if (min_psp_recall is not None) and (recP + 1e-12 < min_psp_recall):
            continue

        if metric == "accuracy":
            s = accuracy_score(y_true, y_pred)
        elif metric == "balanced_accuracy":
            s = balanced_accuracy_score(y_true, y_pred)
        elif metric == "psp_recall":
            s = recP
        elif metric == "f1_macro":
            s = f1_score(y_true, y_pred, average="macro")
        else:
            s = balanced_accuracy_score(y_true, y_pred)

        cand.append((t, s))

    if len(cand) == 0:
        # 제약을 만족하는 임계값이 없다면: PSP recall을 최대화하는 t로 fallback
        for t in ths:
            y_pred = (proba_psp >= t).astype(int)
            recP = recall_score(y_true, y_pred, pos_label=CLASS_TO_IDX["PSP"])
            if recP > best_s:
                best_s, best_t = recP, t
        return float(best_t)

    best_t, _ = max(cand, key=lambda z: z[1])
    return float(best_t)

# ===== SVM 파이프라인/그리드 =====
def build_pipe(use_pca: bool, pca_whiten: bool):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(svd_solver="full", whiten=pca_whiten) if use_pca else "passthrough"),
        ("clf", LinearSVC(class_weight="balanced", dual=False, max_iter=10000)),  # placeholder
    ])

def get_model_grids(use_pca: bool) -> Dict[str, Dict[str, Any]]:
    def pca_grid():
        return {"pca__n_components": [0.90, 0.95]} if use_pca else {}

    # class_weight 사전은 정수 라벨(0:HC, 1:PSP)에 맞춰야 함
    cw_balanced = "balanced"
    cw_psp12 = {CLASS_TO_IDX["HC"]: 1.0, CLASS_TO_IDX["PSP"]: 1.2}
    cw_psp15 = {CLASS_TO_IDX["HC"]: 1.0, CLASS_TO_IDX["PSP"]: 1.5}

    return {
        # 기존 SVM 2종
        "SVM_Linear": {
            **pca_grid(),
            "clf": [LinearSVC(dual=False, max_iter=10000)],
            "clf__C": [0.1, 1, 10],
            "clf__class_weight": [cw_balanced, cw_psp12, cw_psp15],
        },
        "SVM_RBF": {
            **pca_grid(),
            "clf": [SVC(kernel="rbf")],
            "clf__C": [0.1, 0.3, 1, 3, 10, 30, 100],
            "clf__gamma": ["scale", "auto", 0.001, 0.003, 0.01, 0.03, 0.1],
            "clf__class_weight": [cw_balanced, cw_psp12, cw_psp15],
        },
        # 새로 추가: Logistic, LDA (확률 출력에 유리 → 앙상블 적합)
        "Logistic": {
            **pca_grid(),
            "clf": [__import__("sklearn.linear_model").linear_model.LogisticRegression(max_iter=2000)],
            "clf__solver": ["liblinear"],          # 소규모/이진에 안정적
            "clf__C": [0.1, 1, 10],
            "clf__class_weight": [cw_balanced, cw_psp12, cw_psp15],
        },
        "LDA": {
            **pca_grid(),
            "clf": [__import__("sklearn.discriminant_analysis").discriminant_analysis.LinearDiscriminantAnalysis()],
            "clf__solver": ["lsqr"],
            "clf__shrinkage": ["auto"],
        },
    }
    
def has_proba(estimator) -> bool:
    return hasattr(estimator, "predict_proba")

# 교차보정(cv=5) 채택: 작은 데이터에서 과신 완화
def ensure_calibrated(estimator, calibration: str, Xtr2, ytr2):
    needs = (not has_proba(estimator)) or (calibration != "none")
    if not needs:
        fitted = estimator.fit(Xtr2, ytr2)
        return fitted, "none"
    method = calibration if calibration != "none" else "sigmoid"
    cal = CalibratedClassifierCV(estimator=estimator, method=method, cv=5)
    cal.fit(Xtr2, ytr2)
    return cal, method

# =========================
# ENSEMBLE UTILS
# =========================
from typing import Optional

def generate_weight_vectors(k: int, step: float = 0.05, max_candidates: int = 10000):
    steps = int(round(1.0 / step))
    if k == 1:
        yield np.array([1.0], dtype=float); return
    if k == 2:
        for i in range(steps + 1):
            w1 = i * step
            yield np.array([w1, 1.0 - w1], dtype=float)
    else:
        cnt = 0
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                w1 = i * step
                w2 = j * step
                w3 = 1.0 - w1 - w2
                if w3 < -1e-9:
                    continue
                yield np.array([w1, w2, max(0.0, w3)], dtype=float)
                cnt += 1
                if cnt >= max_candidates:
                    return

def _reorder_proba_to_base(p: np.ndarray, classes_src: List[int], classes_base: List[int]) -> np.ndarray:
    """각 모델의 classes 순서를 base에 맞추도록 확률 열을 재정렬"""
    idx_map = [classes_src.index(c) for c in classes_base]
    return p[:, idx_map]

def combine_probas(weight_vec: np.ndarray, probas: List[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(probas[0])
    for w, p in zip(weight_vec, probas):
        out += w * p
    return out

def tune_threshold_with_min_recall(y_true: np.ndarray, p_psp: np.ndarray,
                                   metric: str, min_psp_recall: Optional[float] = None) -> Tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (p_psp >= t).astype(int)  # PSP=1
        if min_psp_recall is not None:
            rec_psp = recall_score(y_true, y_pred, pos_label=CLASS_TO_IDX["PSP"])
            if rec_psp + 1e-12 < min_psp_recall:
                continue
        if metric == "accuracy":
            s = accuracy_score(y_true, y_pred)
        elif metric == "balanced_accuracy":
            s = balanced_accuracy_score(y_true, y_pred)
        elif metric == "f1_macro":
            s = f1_score(y_true, y_pred, average="macro")
        else:
            s = balanced_accuracy_score(y_true, y_pred)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)

class EnsembleManager:
    """
    개별 모델들을 등록(add)한 뒤:
      - 검증셋에서 가중치(soft-vote) + 임계값 동시 탐색(목표 metric 최대)
      - PSP 민감도 하한(min_psp_recall) 강제
      - 테스트 성능 출력 및 저장
    """
    def __init__(self, class_to_idx: Dict[str, int], idx_to_class: Dict[int, str]):
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

    def add(self, name: str, model, classes_: List[int],
            proba_val: np.ndarray, proba_te: np.ndarray):
        self.registry[name] = {
            "model": model, "classes": list(classes_),
            "proba_val": proba_val, "proba_te": proba_te
        }

    def run_softvote(self, yval: np.ndarray, yte: np.ndarray,
                     members: List[str], weight_step: float,
                     tune_metric: str, min_psp_recall: float | None,
                     out_path: str, feat_cfg, meta: Dict[str, Any]):

        assert len(members) >= 2, f"[ENSEMBLE] 멤버 수 부족: {members}"
        for m in members:
            assert m in self.registry, f"[ENSEMBLE] 멤버 '{m}'가 등록되지 않음. 등록된: {list(self.registry.keys())}"

        # 기준 클래스 순서(첫 멤버)
        base_classes = self.registry[members[0]]["classes"]
        idx_psp = base_classes.index(self.class_to_idx["PSP"])

        # 멤버별 확률을 base 순서로 재정렬
        probas_val = []
        probas_te  = []
        for m in members:
            cls_m = self.registry[m]["classes"]
            pv_m  = _reorder_proba_to_base(self.registry[m]["proba_val"], cls_m, base_classes)
            pt_m  = _reorder_proba_to_base(self.registry[m]["proba_te"],  cls_m, base_classes)
            probas_val.append(pv_m)
            probas_te.append(pt_m)

        # 가중치 + 임계값 탐색
        best = {"w": None, "th": 0.5, "score": -1.0}
        for w in generate_weight_vectors(len(members), step=weight_step):
            pv = combine_probas(w, probas_val)
            th, s = tune_threshold_with_min_recall(
                y_true=yval, p_psp=pv[:, idx_psp],
                metric=tune_metric, min_psp_recall=min_psp_recall
            )
            if s > best["score"]:
                best = {"w": w, "th": th, "score": s}

        print(f"[ENSEMBLE] members={members}")
        print(f"[ENSEMBLE] best weights:", {m: float(w) for m,w in zip(members, best["w"])})
        print(f"[ENSEMBLE] tuned threshold (val, {tune_metric} max | PSP≥{min_psp_recall}): {best['th']:.3f} (score={best['score']:.3f})")

        # 테스트 평가
        pte = combine_probas(best["w"], probas_te)
        ypred_thr = (pte[:, idx_psp] >= best["th"]).astype(int)
        acc  = accuracy_score(yte, ypred_thr)
        ba   = balanced_accuracy_score(yte, ypred_thr)
        recP = recall_score(yte, ypred_thr, pos_label=self.class_to_idx["PSP"])
        f1m  = f1_score(yte, ypred_thr, average="macro")
        try:
            ll = log_loss(yte, pte, labels=base_classes)
        except Exception:
            ll = float("nan")

        print(f"[ENSEMBLE RESULT] Test Acc = {acc:.4f}, BA = {ba:.4f}, PSP Recall = {recP:.4f}, F1-macro = {f1m:.4f}, LogLoss = {ll:.4f}")
        print("\n--- Classification Report (thresholded, ENSEMBLE) ---")
        print(classification_report(yte, ypred_thr, labels=base_classes, 
                                    target_names=[self.idx_to_class[i] for i in base_classes], digits=4))
        print("--- Confusion Matrix (thresholded, ENSEMBLE) ---")
        print(confusion_matrix(yte, ypred_thr, labels=base_classes))

        joblib.dump({
            "classes": [self.idx_to_class[i] for i in base_classes],
            "label_map": self.class_to_idx,
            "ensemble": {
                "type": "softvote",
                "members": members,
                "weights": best["w"],
                "threshold": best["th"],
                "tune_metric": tune_metric,
                "min_psp_recall": min_psp_recall,
            },
            "models": {name: self.registry[name]["model"] for name in members},
            "feat_cfg": feat_cfg,
            "meta": meta,
        }, out_path)
        print(f"[OK] 앙상블 모델 저장: {out_path}")

# ===== Main =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./DATA/records")
    parser.add_argument("--pattern", type=str, default="*_clean*.wav")
    parser.add_argument("--use_pca", action="store_true", default=True)
    parser.add_argument("--pca_whiten", action="store_true")
    parser.add_argument("--scoring", type=str, default="balanced_accuracy",
                        choices=["accuracy", "balanced_accuracy"])
    parser.add_argument("--calibration", type=str, default="sigmoid",
                        choices=["none", "sigmoid", "isotonic"])
    parser.add_argument("--threshold_tune", type=str, default="balanced_accuracy",
                        choices=["none", "accuracy", "balanced_accuracy", "psp_recall", "f1_macro"])
    parser.add_argument("--min_psp_recall", type=float, default=None,
                        help="임계값 탐색 시 PSP Recall 하한(e.g., 0.60). 만족 후보 중 metric 최대값 선택.")
    parser.add_argument("--aug_hc", type=int, default=0)
    parser.add_argument("--aug_psp", type=int, default=6)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--max_train_samples", type=int, default=None)

    # --- Ensemble options ---
    parser.add_argument("--ensemble", type=str, default="none",
                        choices=["none", "softvote"],
                        help="none | softvote (확률 가중 평균 앙상블)")
    parser.add_argument("--ensemble_members", type=str, default="SVM_RBF,SVM_Linear",
                        help="콤마로 구분: 예) SVM_RBF,SVM_Linear")
    parser.add_argument("--weight_step", type=float, default=0.05,
                        help="가중치 탐색 간격(0.05 권장; 멤버 2~3개 권장)")

    args = parser.parse_args()

    # 1) 파일
    paths, labels = list_files_by_name(args.data_dir, pattern=args.pattern)
    print(f"[INFO] 총 {len(paths)}개 파일 로드")
    print("[INFO] 라벨 분포:", Counter(labels))

    # 2) 홀드아웃(피험자 기준)
    subjects = np.array([subject_id_from_name(Path(p).name) for p in paths])
    labels_arr = np.array([CLASS_TO_IDX[l] for l in labels], dtype=int)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (tr_idx, te_idx), = outer.split(np.arange(len(paths)), labels_arr, groups=subjects)
    print("[INFO] Outer holdout by subject:", len(tr_idx), "train,", len(te_idx), "test")
    paths_tr = [paths[i] for i in tr_idx]; labels_tr = [labels[i] for i in tr_idx]
    paths_te = [paths[i] for i in te_idx]; labels_te = [labels[i] for i in te_idx]

    # 3) 특징 행렬(학습: 병렬 증강 / 테스트: 기본)
    cfg = FeatConfig(use_pca=args.use_pca)
    Xtr, ytr, subs_tr = build_feature_matrix_aug_per_class_parallel(
        paths_tr, labels_tr, cfg,
        aug_hc=args.aug_hc, aug_psp=args.aug_psp,
        n_jobs=args.n_jobs, preload_audio=True,
        max_train_samples=args.max_train_samples
    )
    def build_feature_matrix(paths: List[str], labels: List[str], cfg: FeatConfig):
        X, y, subs = [], [], []
        it = tqdm(zip(paths, labels), total=len(paths), desc="[Extract] features", ncols=100)
        for p, lab in it:
            ysig = load_audio(p, cfg.sr, cfg.top_db)
            X.append(extract_features(ysig, cfg.sr, cfg))
            y.append(CLASS_TO_IDX[lab]); subs.append(subject_id_from_name(Path(p).name))
        return np.stack(X, axis=0), np.array(y, dtype=int), np.array(subs)
    Xte, yte, subs_te = build_feature_matrix(paths_te, labels_te, cfg)
    print(f"[INFO] Feature shapes — Train: {Xtr.shape}, Test: {Xte.shape}")

    # (NEW) 앙상블 매니저 준비
    ens = EnsembleManager(CLASS_TO_IDX, IDX_TO_CLASS)

    # 4) CV 설정
    n_groups = np.unique(subs_tr).size
    n_splits = min(5, n_groups) if n_groups >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits)

    # 5) 검증 분할(보정/임계값 튜닝용)
    cal_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
    (tr2_idx, val_idx), = cal_split.split(Xtr, ytr, groups=subs_tr)
    Xtr2, ytr2 = Xtr[tr2_idx], ytr[tr2_idx]
    Xval, yval = Xtr[val_idx], ytr[val_idx]

    # 6) SVM 모델들: 탐색→보정(cv=5)→튜닝(제약)→평가→저장 (+ 앙상블 등록)
    pipe_base = build_pipe(args.use_pca, args.pca_whiten)
    svm_grids = get_model_grids(args.use_pca)

    summary_rows = []
    for model_name, param_grid in svm_grids.items():
        print("\n" + "="*80)
        print(f"[MODEL] {model_name}")
        print("="*80)

        gs = GridSearchCV(
            estimator=pipe_base,
            param_grid=param_grid,
            scoring=args.scoring,
            cv=gkf,
            n_jobs=-1,
            verbose=1,
            refit=True
        )
        gs.fit(Xtr, ytr, groups=subs_tr)
        print("[TUNE] best_params:", gs.best_params_)
        print(f"[TUNE] best_cv_{args.scoring}:", gs.best_score_)

        estimator = gs.best_estimator_
        final_model, used_cal = ensure_calibrated(estimator, args.calibration, Xtr2, ytr2)

        # 임계값 튜닝(검증, 제약식)
        proba_val = final_model.predict_proba(Xval)
        cls_order = list(final_model.classes_)
        idx_psp = cls_order.index(CLASS_TO_IDX["PSP"])
        t_opt = tune_threshold(yval, proba_val[:, idx_psp],
                               metric=args.threshold_tune,
                               min_psp_recall=args.min_psp_recall)
        print(f"[THRESH] tuned threshold (metric={args.threshold_tune}, min_PSP_recall={args.min_psp_recall}): {t_opt:.3f}")

        # 테스트 평가
        proba_te = final_model.predict_proba(Xte)
        ypred_thr = (proba_te[:, idx_psp] >= t_opt).astype(int)  # PSP=1
        acc  = accuracy_score(yte, ypred_thr)
        ba   = balanced_accuracy_score(yte, ypred_thr)
        recP = recall_score(yte, ypred_thr, pos_label=CLASS_TO_IDX["PSP"])
        f1m  = f1_score(yte, ypred_thr, average="macro")
        try:
            ll = log_loss(yte, proba_te, labels=final_model.classes_)
        except Exception:
            ll = float("nan")

        names_by_model = [IDX_TO_CLASS[i] for i in cls_order]
        print(f"[RESULT] Test Acc = {acc:.4f}, BA = {ba:.4f}, PSP Recall = {recP:.4f}, F1-macro = {f1m:.4f}, LogLoss = {ll:.4f}")
        print("\n--- Classification Report (thresholded) ---")
        print(classification_report(yte, ypred_thr, labels=cls_order, target_names=names_by_model, digits=4))
        print("--- Confusion Matrix (thresholded) ---")
        print(confusion_matrix(yte, ypred_thr, labels=cls_order))
        print("\n--- Per-class 평균 예측 확률(전체 표본 기준) ---")
        for name, m in zip(names_by_model, np.mean(proba_te, axis=0)):
            print(f" {name}: {m:.3f}")

        # 저장
        if model_name == "SVM_Linear":
            suffix = "SVM_Linear_ENH"
        elif model_name == "SVM_RBF":
            suffix = "SVM_RBF_ENH"
        elif model_name == "Logistic":
            suffix = "Logistic_ENH"
        elif model_name == "LDA":
            suffix = "LDA_ENH"
        else:
            suffix = model_name.replace(" ", "_") + "_ENH"
        out_path = f"./voice_model_HC_vs_PSP__{suffix}.joblib"
        joblib.dump({
            "classes": names_by_model,
            "label_map": CLASS_TO_IDX,
            "model": final_model,
            "feat_cfg": cfg,
            "pca_used": args.use_pca,
            "pca_whiten": args.pca_whiten,
            "scoring": args.scoring,
            "calibration": used_cal,
            "decision_threshold": t_opt,
            "positive_class": "PSP",
            "aug_hc": args.aug_hc,
            "aug_psp": args.aug_psp,
            "n_jobs_aug": args.n_jobs,
            "max_train_samples": args.max_train_samples,
            "best_params": gs.best_params_,
            "cv_best_score": gs.best_score_,
            "model_name": model_name,
            "feature_set": "ENH(MFCC+d1+d2 + spec + ZCR + F0/RMS + HNR + CPP + F1-3 + dur/skew/kurt)"
        }, out_path)
        print(f"[OK] 모델 저장: {out_path}")

        summary_rows.append({
            "Model": model_name,
            "CV_best_score": float(gs.best_score_),
            "Calib": used_cal,
            "Thresh": float(t_opt),
            "Acc_test": float(acc),
            "BA_test": float(ba),
            "PSP_Recall_test": float(recP),
            "F1_macro_test": float(f1m),
            "LogLoss_test": float(ll),
            "Saved_to": out_path
        })

        # (NEW) 앙상블 등록: 검증/테스트 확률 & 클래스 순서
        ens.add(model_name, final_model, list(final_model.classes_), proba_val, proba_te)

    # 요약 CSV
    if len(summary_rows) > 0:
        csv_path = "./model_compare_summary_SVM_ENH.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows: writer.writerow(r)
        print(f"\n[SUMMARY] 저장: {csv_path}")
        print("== 결과 요약 (SVM 전용, Enhanced features) ==")
        for r in summary_rows:
            print(f"{r['Model']}: BA={r['BA_test']:.4f}, Acc={r['Acc_test']:.4f}, PSP_Rec={r['PSP_Recall_test']:.4f}, F1m={r['F1_macro_test']:.4f} | Calib={r['Calib']} | Th={r['Thresh']:.3f}")

    # =========================
    # ENSEMBLE 실행 (옵션)
    # =========================
    if args.ensemble == "softvote":
        members = [m.strip() for m in args.ensemble_members.split(",") if m.strip()]
        ens.run_softvote(
            yval=yval, yte=yte,
            members=members,
            weight_step=args.weight_step,
            tune_metric=("balanced_accuracy" if args.threshold_tune == "none" else args.threshold_tune),
            min_psp_recall=(args.min_psp_recall if args.threshold_tune != "none" else None),
            out_path="./voice_model_HC_vs_PSP__ENS_SoftVote_ENH.joblib",
            feat_cfg=cfg,
            meta={
                "pca_whiten": args.pca_whiten,
                "scoring": args.scoring,
                "calibration": args.calibration
            }
        )
