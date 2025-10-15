import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ------------------------------------------------------------
# Utility: permutation-invariant RMSE for two targets
# ------------------------------------------------------------
def permutation_invariant_rmse(true_angles, est_angles):
    t = np.deg2rad(true_angles)
    e = np.deg2rad(est_angles)
    err1 = np.mean((t - e)**2)
    err2 = np.mean((t - e[::-1])**2)
    return np.sqrt(min(err1, err2))

# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------
c = 3e8                 # speed of light (m/s)
N_f = 40                # number of frequencies
fmin, fmax = 200e9, 800e9
fn = np.linspace(fmin, fmax, N_f)
L, b = 25e-3, 1e-3      # LWA dimensions
d = 0.5 * c / np.median(fn)  # ULA element spacing

phis = np.linspace(0, 40, 41)  # search grid (°)
SNR_dB = np.linspace(-10, 10, 5)
SNR = 10**(SNR_dB/10)

num_MC = 150
ne = 5                 # ULA elements
delta = 1e-6           # FD step for CRB
T_snapshots = 50       # number of snapshots for all methods
K = 2                  # number of targets
angle_thresh = 6       # minimum angular separation

# ------------------------------------------------------------
# LWA steering vector and derivative
# ------------------------------------------------------------
def lwa_signal(theta):
    G = np.array([
        np.sinc((2*np.pi*f/c * np.sqrt(1 - (c/(2*b*f))**2)
                 - 2*np.pi*f/c * np.cos(theta)) * L/2)
        for f in fn
    ])
    return G / np.linalg.norm(G)

def lwa_signal_derivative(theta):
    u = (2*np.pi*fn/c * np.sqrt(1 - (c/(2*b*fn))**2)
         - 2*np.pi*fn/c * np.cos(theta)) * L/2
    G = np.sinc(u)
    normG = np.linalg.norm(G)
    dsinc_du = np.zeros_like(u)
    nz = u!=0
    dsinc_du[nz] = (np.pi*u[nz]*np.cos(np.pi*u[nz]) - np.sin(np.pi*u[nz]))/(np.pi*u[nz]**2)
    du = 2*np.pi*fn/c * np.sin(theta) * L/2
    dG = dsinc_du * du
    dnorm = (G.conj() @ dG) / normG
    return dG / normG - G * dnorm / (normG**2)

# Precompute LWA cache and MUSIC steering
theta_rad = np.deg2rad(phis)
lwa_cache = np.stack([lwa_signal(th) for th in theta_rad], axis=1)
a_music = lwa_cache / np.linalg.norm(lwa_cache, axis=0, keepdims=True)

# ------------------------------------------------------------
# MUSIC estimator for two targets
# ------------------------------------------------------------
def estimate_angles_music(Y, A, grid):
    R = Y @ Y.conj().T / Y.shape[1]
    _, U = np.linalg.eigh(R)
    Un = U[:, :-K]
    spec = np.array([1/np.real(a.conj() @ Un @ Un.conj().T @ a) for a in A.T])
    peaks, _ = find_peaks(spec, distance=int(angle_thresh))
    if len(peaks) >= K:
        top = peaks[np.argsort(spec[peaks])[-K:]]
        return sorted(grid[top])
    return [0]*K

# ------------------------------------------------------------
# Containers for results
# ------------------------------------------------------------
errs_ml = np.zeros((num_MC, len(SNR_dB)))
errs_corr = np.zeros((num_MC, len(SNR_dB)))
errs_music = np.zeros((num_MC, len(SNR_dB)))
errs_ula = np.zeros((num_MC, len(SNR_dB)))
crb_num = np.zeros((num_MC, len(SNR_dB)))
crb_ana = np.zeros((num_MC, len(SNR_dB)))

# ------------------------------------------------------------
# Monte Carlo simulation
# ------------------------------------------------------------
for mc in range(num_MC):
    while True:
        true_angles = np.sort(np.random.uniform(0, 40, K))
        if true_angles[1] - true_angles[0] >= angle_thresh:
            break
    G_comb = sum(lwa_signal(np.deg2rad(a)) for a in true_angles)
    G_comb /= np.linalg.norm(G_comb)

    for i_snr, snr in enumerate(SNR):
        sigma2 = 1 / snr
        noise = np.sqrt(sigma2/2)*(np.random.randn(N_f, T_snapshots)+1j*np.random.randn(N_f, T_snapshots))
        Y = G_comb[:, None] + noise

        # ML
        res_ml = np.zeros(len(phis))
        for ti in range(T_snapshots):
            y = Y[:, ti]
            for i,p1 in enumerate(phis):
                for j,p2 in enumerate(phis[i+1:], start=i+1):
                    if p2-p1 < angle_thresh: continue
                    s = lwa_cache[:, i] + lwa_cache[:, j]
                    res_ml[i] += np.linalg.norm(s - y)**2
                    res_ml[j] += np.linalg.norm(s - y)**2
        est_ml = sorted(phis[np.argsort(res_ml)[:K]])

        # Correlation
        corr = np.zeros(len(phis))
        for ti in range(T_snapshots):
            y = Y[:, ti]
            corr += np.abs([np.real(np.vdot(y, lwa_cache[:, idx])) for idx in range(len(phis))])
        est_corr = sorted(phis[np.argsort(-corr)[:K]])

        # LWA-MUSIC (angle estimates only, no spectrum plot)
        est_m = estimate_angles_music(Y, a_music, phis)

        # ULA-BF
        X_arr = np.zeros((ne, N_f, T_snapshots), complex)
        for fi,f in enumerate(fn):
            k = 2 * np.pi * f / c
            for ti in range(T_snapshots):
                for ang in true_angles:
                    sv = np.exp(-1j * k * d * np.arange(ne) * np.sin(np.deg2rad(ang)))
                    X_arr[:, fi, ti] += sv + np.sqrt(sigma2/2)*(np.random.randn(ne)+1j*np.random.randn(ne))
        bf_spec = np.zeros(len(phis))
        for ti in range(T_snapshots):
            Xi = X_arr[:, :, ti]
            for idx,p in enumerate(phis):
                a_vec = np.exp(-1j * (2*np.pi*fn/c)[:, None] * d * np.arange(ne) * np.sin(np.deg2rad(p)))
                bf_spec[idx] += np.sum(np.abs(np.vdot(a_vec, Xi.T))**2)
        est_ula = sorted(phis[np.argsort(-bf_spec)[:K]])

        errs_ml[mc, i_snr]    = permutation_invariant_rmse(true_angles, est_ml)
        errs_corr[mc, i_snr]  = permutation_invariant_rmse(true_angles, est_corr)
        errs_music[mc, i_snr] = permutation_invariant_rmse(true_angles, est_m)
        errs_ula[mc, i_snr]   = permutation_invariant_rmse(true_angles, est_ula)

        # Numeric CRB
        th1, th2 = np.deg2rad(true_angles)
        ds1 = (lwa_signal(th1+delta) - lwa_signal(th1-delta)) / (2*delta)
        ds2 = (lwa_signal(th2+delta) - lwa_signal(th2-delta)) / (2*delta)
        F_num = (1/sigma2)*np.array([[ds1@ds1, ds1@ds2], [ds2@ds1, ds2@ds2]])
        crb_num[mc, i_snr] = np.sqrt(np.trace(np.linalg.inv(F_num))/2)

        # Analytic CRB
        d1 = lwa_signal_derivative(th1)
        d2 = lwa_signal_derivative(th2)
        F_ana = (1/sigma2)*np.array([[np.real(d1.conj()@d1), np.real(d1.conj()@d2)],
                                     [np.real(d2.conj()@d1), np.real(d2.conj()@d2)]])
        crb_ana[mc, i_snr] = np.sqrt(np.trace(np.linalg.inv(F_ana))/2)

# Performance plot (RMSE + CRB)
m_ml   = errs_ml.mean(0)
m_corr = errs_corr.mean(0)
m_m    = errs_music.mean(0)
m_ula  = errs_ula.mean(0)
m_ca   = crb_ana.mean(0)

plt.figure(figsize=(10,6))
plt.plot(SNR_dB, m_ml,   '-o',  label='Maximum Likelihood')
plt.plot(SNR_dB, m_corr, '--s',  label='Correlation-Based')
plt.plot(SNR_dB, m_m,    '-.^', label='LWA MUSIC')
plt.plot(SNR_dB, m_ula,  ':x',  label='ULA BF')
plt.plot(SNR_dB, m_ca,   ':v',  label='CRB')
plt.xlabel('SNR (dB)')
plt.ylabel('RMSE (rad)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
