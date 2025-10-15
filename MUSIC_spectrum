#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LWA MUSIC Pseudospectrum (Two Users) — Frequency-Manifold MUSIC

- Treats frequency bins as "sensors" of a Leaky-Wave Antenna (LWA)
- Simulates T time snapshots so the sample covariance has rank K=2
- Computes MUSIC pseudospectrum over angle and marks:
    * TRUE angles (dashed lines)
    * DETECTED peaks (dots + labels)

Dependencies: numpy, matplotlib
Run: python lwa_music_two_users.py
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters (edit freely)
# =========================
c = 3e8                     # speed of light [m/s]
N_f = 40                    # number of frequency bins
fmin, fmax = 200e9, 800e9   # frequency range [Hz]
freqs = np.linspace(fmin, fmax, N_f)

L = 25e-3                   # slit length [m]
b = 1e-3                    # plate separation [m]

true_angles_deg = [12.0, 26.0]  # two users (degrees)
K = 2                        # number of sources (fixed at 2 here)
T = 200                      # snapshots (time frames)
SNR_dB = 10.0                # SNR per frequency "sensor" [dB]
diag_loading = 1e-6          # covariance stabilizer

angle_grid = np.linspace(0.0, 40.0, 40)  # fine grid (0.025° steps)
min_separation_deg = 3.0                   # min distance between selected peaks

# =========================
# LWA steering & simulation
# =========================
def lwa_steering_freq(theta_rad: float,
                      freqs: np.ndarray,
                      L: float,
                      b: float,
                      c: float = 3e8) -> np.ndarray:
    """
    Frequency-manifold steering vector g(theta) of length N_f for LWA.
    Uses the sinc-based model, normalized to unit 2-norm.
    """
    k = 2 * np.pi * freqs / c
    k_beta = k * np.sqrt(1.0 - (c / (2.0 * b * freqs))**2)   # simple guided wavenumber model
    u = (k_beta - k * np.cos(theta_rad)) * (L / 2.0)
    # np.sinc(x) = sin(pi x)/(pi x); divide u by pi to match this definition
    g = np.sinc(u / np.pi)
    nrm = np.linalg.norm(g)
    return g if nrm == 0 else g / nrm

def simulate_lwa_two_users(freqs: np.ndarray,
                           thetas_deg: list[float],
                           T: int,
                           SNR_dB: float,
                           L: float,
                           b: float,
                           c: float = 3e8) -> np.ndarray:
    """
    Simulate T snapshots: y_t = Σ_k s_k[t] * g(theta_k) + w_t
    Returns Y of shape (N_f, T).
    """
    N_f = len(freqs)
    Y = np.zeros((N_f, T), dtype=complex)
    SNR = 10.0**(SNR_dB / 10.0)
    noise_var = 1.0 / SNR  # unit signal variance per user

    Gs = [lwa_steering_freq(np.deg2rad(th), freqs, L, b, c) for th in thetas_deg]  # (N_f,) each
    S = (np.random.randn(len(thetas_deg), T) + 1j*np.random.randn(len(thetas_deg), T)) / np.sqrt(2)

    for t in range(T):
        y = np.zeros(N_f, dtype=complex)
        for k in range(len(thetas_deg)):
            y += S[k, t] * Gs[k]
        noise = np.sqrt(noise_var/2) * (np.random.randn(N_f) + 1j*np.random.randn(N_f))
        Y[:, t] = y + noise
    return Y

# =========================
# MUSIC & peak picking
# =========================
def music_pseudospectrum_lwa(Y: np.ndarray,
                             freqs: np.ndarray,
                             angles_deg: np.ndarray,
                             L: float,
                             b: float,
                             K: int,
                             diag_loading: float = 1e-6,
                             c: float = 3e8) -> np.ndarray:
    """
    Compute LWA MUSIC pseudospectrum P(θ) = 1 / || U_n^H g(θ) ||^2.
    Y: (N_f, T) snapshots → R = YY^H/T (N_f x N_f) covariance.
    """
    # Sample covariance across time
    R = (Y @ Y.conj().T) / max(Y.shape[1], 1)
    R = (R + R.conj().T) / 2.0  # Hermitian symmetrization
    R += diag_loading * np.eye(R.shape[0])

    # Eigendecomposition: smallest eigenvalues = noise subspace
    w, V = np.linalg.eigh(R)     # ascending
    n_noise = max(R.shape[0] - K, 1)
    Un = V[:, :n_noise]          # (N_f, N_f - K)

    # Scan grid
    P = np.zeros_like(angles_deg, dtype=float)
    for i, ang in enumerate(angles_deg):
        g = lwa_steering_freq(np.deg2rad(ang), freqs, L, b, c).astype(complex)
        denom = np.linalg.norm(Un.conj().T @ g)**2
        P[i] = 1.0 / max(denom, 1e-16)

    # Normalize
    P -= P.min()
    if P.max() > 0:
        P /= P.max()
    return P

def pick_top_k_peaks(y: np.ndarray, k: int = 2, min_sep_bins: int = 1) -> np.ndarray:
    """
    SciPy-free peak picker: local maxima → greedy top-k with minimum separation (in bins).
    Returns indices of chosen peaks (sorted).
    """
    if y.size < 3:
        return np.array([], dtype=int)
    cand = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
    if cand.size == 0:
        return np.array([], dtype=int)

    # Sort by height desc
    cand = cand[np.argsort(y[cand])[::-1]]
    chosen = []
    for idx in cand:
        if all(abs(idx - c) >= min_sep_bins for c in chosen):
            chosen.append(idx)
            if len(chosen) == k:
                break
    return np.array(sorted(chosen), dtype=int)

# =========================
# Main
# =========================
def main():
    # Simulate snapshots
    Y = simulate_lwa_two_users(freqs, true_angles_deg, T, SNR_dB, L, b, c)

    # MUSIC spectrum
    P = music_pseudospectrum_lwa(Y, freqs, angle_grid, L, b, K, diag_loading, c)

    # Detect top-2 peaks with min separation
    step_deg = angle_grid[1] - angle_grid[0]
    min_sep_bins = max(1, int(np.round(min_separation_deg / step_deg)))
    peak_idx = pick_top_k_peaks(P, k=2, min_sep_bins=min_sep_bins)
    detected = np.sort(angle_grid[peak_idx]) if peak_idx.size > 0 else np.array([])

    print(f"True angles (deg): {np.round(true_angles_deg, 2)}")
    if detected.size == 2:
        print(f"LWA MUSIC detected peaks (deg): {np.round(detected, 2)}")
    else:
        print("LWA MUSIC detected peaks: <not enough peaks detected>")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(angle_grid, P, linewidth=2, label='LWA MUSIC pseudospectrum')
    for i, ta in enumerate(true_angles_deg):
        plt.axvline(ta, linestyle='--', alpha=0.7, label=f'True angle {ta:.2f}°' if i == 0 else None)
    if peak_idx.size > 0:
        plt.plot(detected, P[peak_idx], 'o', markersize=8, label='Detected peaks')
        for pi, a in zip(peak_idx, detected):
            plt.annotate(f'{a:.2f}°', (a, P[pi]), xytext=(0, 6),
                         textcoords='offset points', ha='center', fontsize=10)
    plt.title(f'LWA MUSIC Pseudospectrum (K=2, T={T}, SNR={SNR_dB:.1f} dB)')
    plt.xlabel('Angle (deg)')
    plt.ylabel('Normalized pseudospectrum')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
