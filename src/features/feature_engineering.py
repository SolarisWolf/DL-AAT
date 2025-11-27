"""
Feature engineering utilities for power grid fault detection.
Includes:
- Symmetrical components (positive/negative/zero sequence) for 3-phase signals
- RMS over windows
- THD (Total Harmonic Distortion) over windows

Works with windows shaped as (T, 6) where columns are [Va,Vb,Vc,Ia,Ib,Ic].
Also provides a small CLI to compute features from an existing npz dataset.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import argparse
import json


def _symmetrical_components_three_phase(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute symmetrical components for a 3-phase complex phasor array.
    Input v shape: (..., 3), ordered (A,B,C). Returns (V0, V1, V2) arrays.
    """
    # Fortescue operator a = e^{j 2pi/3}
    a = np.exp(2j * np.pi / 3)
    A = np.array([
        [1, 1, 1],
        [1, a, a**2],
        [1, a**2, a]
    ], dtype=complex) / 3.0
    # v expected last dim of size 3
    v = np.asarray(v, dtype=complex)
    v_stack = np.moveaxis(v, -1, -1)
    # Apply transform: [V0, V1, V2]^T = A @ [Va,Vb,Vc]^T
    V = np.tensordot(v_stack, A.T, axes=([v_stack.ndim - 1],[0]))
    V0 = V[..., 0]
    V1 = V[..., 1]
    V2 = V[..., 2]
    return V0, V1, V2


def symmetrical_components(Va: np.ndarray, Vb: np.ndarray, Vc: np.ndarray) -> dict:
    """Convenience wrapper returning magnitudes of sequence components."""
    V0, V1, V2 = _symmetrical_components_three_phase(np.stack([Va, Vb, Vc], axis=-1))
    return {
        'seq0_mag': np.abs(V0),
        'seq1_mag': np.abs(V1),
        'seq2_mag': np.abs(V2),
    }


def window_rms(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.square(x))))


def thd(signal: np.ndarray, sample_rate: float) -> float:
    """
    Compute Total Harmonic Distortion for a real signal.
    THD = sqrt(sum_{h>1} P_h) / P_1, using magnitude spectrum peak as fundamental.
    """
    x = np.asarray(signal)
    n = len(x)
    if n < 8:
        return 0.0
    # Hann window to reduce leakage
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    mag = np.abs(X)
    # Find fundamental as largest non-DC bin
    if mag.shape[0] <= 1:
        return 0.0
    fundamental_idx = np.argmax(mag[1:]) + 1
    fundamental = mag[fundamental_idx]
    if fundamental <= 1e-12:
        return 0.0
    harmonics_power = np.sum(np.square(mag)) - (mag[0]**2 + fundamental**2)
    thd_val = np.sqrt(max(harmonics_power, 0.0)) / fundamental
    return float(thd_val)


def compute_window_features(window: np.ndarray, sample_rate: float = 1000.0) -> dict:
    """
    Compute features for a single window (T,6): Va,Vb,Vc,Ia,Ib,Ic.
    Returns a flat dict with RMS (V/I), THD (V/I), and sequence magnitudes (V only).
    """
    assert window.ndim == 2 and window.shape[1] == 6, "window must be (T,6)"
    Va, Vb, Vc, Ia, Ib, Ic = [window[:, i] for i in range(6)]

    feats = {}
    # RMS
    feats['Vrms_A'] = window_rms(Va)
    feats['Vrms_B'] = window_rms(Vb)
    feats['Vrms_C'] = window_rms(Vc)
    feats['Irms_A'] = window_rms(Ia)
    feats['Irms_B'] = window_rms(Ib)
    feats['Irms_C'] = window_rms(Ic)

    # THD
    feats['THD_VA'] = thd(Va, sample_rate)
    feats['THD_VB'] = thd(Vb, sample_rate)
    feats['THD_VC'] = thd(Vc, sample_rate)
    feats['THD_IA'] = thd(Ia, sample_rate)
    feats['THD_IB'] = thd(Ib, sample_rate)
    feats['THD_IC'] = thd(Ic, sample_rate)

    # Symmetrical components (voltage only)
    seq = symmetrical_components(Va + 0j, Vb + 0j, Vc + 0j)
    feats['Vseq0_mag'] = float(np.mean(seq['seq0_mag']))
    feats['Vseq1_mag'] = float(np.mean(seq['seq1_mag']))
    feats['Vseq2_mag'] = float(np.mean(seq['seq2_mag']))

    return feats


def compute_dataset_features(npz_path: str, out_path: str | None = None, sample_rate: float = 1000.0):
    data = np.load(npz_path, allow_pickle=True)
    X = data['X']  # (N,T,6)
    y = data['y']  # (N,)
    meta = json.loads(str(data['metadata'])) if 'metadata' in data else {}

    rows = []
    for i in range(X.shape[0]):
        feats = compute_window_features(X[i], sample_rate)
        feats['label'] = int(y[i])
        rows.append(feats)

    # Save as npz (features matrix + labels + columns)
    cols = list(rows[0].keys())
    feat_mat = np.array([[r[c] for c in cols] for r in rows], dtype=float)
    if out_path is None:
        out_path = str(Path(npz_path).with_suffix('').as_posix()) + "_features.npz"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=feat_mat, y=y, columns=np.array(cols, dtype=object), metadata=np.array([meta], dtype=object))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute engineered features from dataset npz")
    parser.add_argument('--data', required=True, help='Path to dataset .npz (with X,y)')
    parser.add_argument('--out', default=None, help='Output path for features .npz')
    parser.add_argument('--fs', type=float, default=1000.0, help='Sampling rate (Hz)')
    args = parser.parse_args()

    out = compute_dataset_features(args.data, args.out, args.fs)
    print(f"Features saved to: {out}")
