import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Utility: RMSE for one target
# ------------------------------------------------------------
def rmse_single(true_deg, est_deg):
    return np.abs(np.deg2rad(true_deg) - np.deg2rad(est_deg))

# ------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------
c = 3e8                 # speed of light (m/s)
N_f = 40                # number of frequencies
fmin, fmax = 200e9, 800e9
fn = np.linspace(fmin, fmax, N_f)
L, b = 25e-3, 1e-3      # LWA dimensions
d = 0.5 * c / np.median(fn)  # ULA element spacing

phis = np.linspace(0, 40, 41)     # search grid (°)
SNR_dB = np.linspace(-10, 10, 5)
SNR = 10**(SNR_dB/10)

num_MC = 150
ne = 5                   # ULA elements
delta = 1e-6             # FD step for CRB
T_snapshots = 50         # number of snapshots for all methods
K = 1                    # one target now!

# ------------------------------------------------------------
# LWA steering vector and derivative (freq-manifold)
# ------------------------------------------------------------
def lwa_signal(theta):
    """Normalized LWA response across frequency for azimuth theta [rad]."""
    G = np.array([
        np.sinc((2*np.pi*f/c * np.sqrt(1 - (c/(2*b*f))**2)
                 - 2*np.pi*f/c * np.cos(theta)) * L/2)
        for f in fn
    ])
    n = np.linalg.norm(G)
    return G / n if n > 0 else G

def lwa_signal_derivative(theta):
    """Analytic derivative of normalized LWA response wrt theta [rad]."""
    u = (2*np.pi*fn/c * np.sqrt(1 - (c/(2*b*fn))**2)
         - 2*np.pi*fn/c * np.cos(theta)) * L/2
    G = np.sinc(u)
    normG = np.linalg.norm(G) + 1e-16
    dsinc_du = np.zeros_like(u)
    nz = u != 0
    dsinc_du[nz] = (np.pi*u[nz]*np.cos(np.pi*u[nz]) - np.sin(np.pi*u[nz])) / (np.pi*u[nz]**2)
    du = 2*np.pi*fn/c * np.sin(theta) * L/2
    dG = dsinc_du * du
    dnorm = (G.conj() @ dG) / normG
    return dG / normG - G * dnorm / (normG**2)

# Precompute LWA cache and MUSIC steering
theta_rad_grid = np.deg2rad(phis)
lwa_cache = np.stack([lwa_signal(th) for th in theta_rad_grid], axis=1)  # (N_f, |grid|)
a_music = lwa_cache / (np.linalg.norm(lwa_cache, axis=0, keepdims=True) + 1e-16)

# ------------------------------------------------------------
# MUSIC estimator (K=1) on LWA frequency-manifold
# ------------------------------------------------------------
def estimate_angle_music(Y, A, grid_deg):
    """
    Y: (N_f, T) snapshots for LWA (freq sensors, time snapshots)
    A: (N_f, |grid|) steering per grid angle (already normalized)
    """
    R = (Y @ Y.conj().T) / Y.shape[1]          # (N_f, N_f)
    R = (R + R.conj().T) / 2.0                 # Hermitian symmetrization
    w, U = np.linalg.eigh(R)                   # ascending eigenvalues
    Un = U[:, :-K] if K < U.shape[1] else U*0  # noise subspace (all but K largest)
    # P(theta) = 1 / || U_n^H a(theta) ||^2
    spec = np.array([1.0 / max(np.linalg.norm(Un.conj().T @ a)**2, 1e-16) for a in A.T])
    # argmax → estimated angle
    idx = int(np.argmax(spec))
    return float(grid_deg[idx])

# ------------------------------------------------------------
# Containers for results
# ------------------------------------------------------------
errs_ml   = np.zeros((num_MC, len(SNR_dB)))
errs_corr = np.zeros((num_MC, len(SNR_dB)))
errs_music= np.zeros((num_MC, len(SNR_dB)))
errs_ula  = np.zeros((num_MC, len(SNR_dB)))
crb_num   = np.zeros((num_MC, len(SNR_dB)))
crb_ana   = np.zeros((num_MC, len(SNR_dB)))

# ------------------------------------------------------------
# Monte Carlo simulation
# ------------------------------------------------------------
for mc in range(num_MC):
    true_angle = float(np.random.uniform(0, 40))   # one user
    G_true = lwa_signal(np.deg2rad(true_angle))    # (N_f,)
    for i_snr, snr in enumerate(SNR):
        sigma2 = 1 / snr

        # LWA snapshots: y_t = g(theta) + w_t
        noise = np.sqrt(sigma2/2) * (np.random.randn(N_f, T_snapshots) + 1j*np.random.randn(N_f, T_snapshots))
        Y = G_true[:, None] + noise  # (N_f, T)

        # ---------- Maximum Likelihood (grid search) ----------
        # For K=1, just pick phi minimizing ||g(phi)-y||^2 aggregated over snapshots
        cost = np.zeros(len(phis))
        for ti in range(T_snapshots):
            y = Y[:, ti]
            # accumulate squared error per grid point
            diff = lwa_cache - y[:, None]                # (N_f, |grid|)
            cost += np.sum(np.abs(diff)**2, axis=0)
        est_ml = float(phis[np.argmin(cost)])

        # ---------- Correlation-based (beam sweep on frequency-manifold) ----------
        # Sum correlations across snapshots and pick argmax
        corr = np.zeros(len(phis))
        for ti in range(T_snapshots):
            y = Y[:, ti]
            corr += np.abs(np.real(np.conj(lwa_cache).T @ y))  # (|grid|,)
        est_corr = float(phis[np.argmax(corr)])

        # ---------- LWA-MUSIC (frequency-manifold) ----------
        est_m = estimate_angle_music(Y, a_music, phis)

        # ---------- ULA Beamforming baseline ----------
        # simulate ULA snapshots independently (wideband, single source per snapshot)
        X_arr = np.zeros((ne, N_f, T_snapshots), complex)
        for fi, f in enumerate(fn):
            k = 2 * np.pi * f / c
            a_true = np.exp(-1j * k * d * np.arange(ne) * np.sin(np.deg2rad(true_angle)))
            for ti in range(T_snapshots):
                X_arr[:, fi, ti] = a_true + np.sqrt(sigma2/2) * (np.random.randn(ne) + 1j*np.random.randn(ne))

        bf_spec = np.zeros(len(phis))
        # incoherent sum across freqs and snapshots
        for ti in range(T_snapshots):
            Xi = X_arr[:, :, ti]  # (ne, N_f)
            for idx, p in enumerate(phis):
                a_vec = np.exp(-1j * (2*np.pi*fn/c)[:, None] * d * np.arange(ne) * np.sin(np.deg2rad(p)))  # (N_f, ne)
                # matched filter power across freqs
                bf_spec[idx] += np.sum(np.abs(a_vec.conj() @ Xi)**2)
        est_ula = float(phis[np.argmax(bf_spec)])

        # ---------- Errors ----------
        errs_ml[mc, i_snr]    = rmse_single(true_angle, est_ml)
        errs_corr[mc, i_snr]  = rmse_single(true_angle, est_corr)
        errs_music[mc, i_snr] = rmse_single(true_angle, est_m)
        errs_ula[mc, i_snr]   = rmse_single(true_angle, est_ula)

        # ---------- CRB (numeric & analytic) ----------
        th = np.deg2rad(true_angle)
        # numeric
        ds = (lwa_signal(th+delta) - lwa_signal(th-delta)) / (2*delta)
        F_num = (1/sigma2) * (ds @ ds.conj()).real  # scalar
        crb_num[mc, i_snr] = 1.0 / max(F_num, 1e-16)

        # analytic
        d_th = lwa_signal_derivative(th)
        F_ana = (1/sigma2) * (d_th.conj() @ d_th).real  # scalar
        crb_ana[mc, i_snr] = 1.0 / max(F_ana, 1e-16)

# ------------------------------------------------------------
# Aggregate and plot (RMSE in radians; CRB is std = sqrt(variance))
# ------------------------------------------------------------
m_ml    = errs_ml.mean(0)
m_corr  = errs_corr.mean(0)
m_music = errs_music.mean(0)
m_ula   = errs_ula.mean(0)
# Convert CRB variance to std (radians)
m_crb_num = np.sqrt(crb_num.mean(0))
m_crb_ana = np.sqrt(crb_ana.mean(0))

plt.figure(figsize=(10,6))
plt.plot(SNR_dB, m_ml,    '-o',  label='ML (LWA freq-manifold)')
plt.plot(SNR_dB, m_corr,  '--s', label='Correlation (LWA)')
plt.plot(SNR_dB, m_music, '-.^', label='LWA MUSIC')
plt.plot(SNR_dB, m_ula,   ':x',  label='ULA BF baseline')
plt.plot(SNR_dB, m_crb_ana, ':v', label='CRB (analytic, std)')
plt.xlabel('SNR (dB)')
plt.ylabel('RMSE / Std (radians)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
