import numpy as np
import matplotlib.pyplot as plt


def permutation_invariant_rmse(true_angles, est_angles):
    # For one target, this returns the absolute error.
    true_angles = np.array(true_angles) * np.pi / 180
    est_angles = np.array(est_angles) * np.pi / 180
    err1 = np.mean((true_angles - est_angles) ** 2)
    err2 = np.mean((true_angles - est_angles[::-1]) ** 2)
    return np.sqrt(min(err1, err2))


# Constants and Parameters
c = 3e8  # Speed of light (m/s)
N_f = 40  # Number of frequencies
fmin = 200e9  # Minimum frequency (Hz)
fmax = 800e9  # Maximum frequency (Hz)
fn = np.linspace(fmin, fmax, N_f)
L = 25e-3  # Fixed slit length (meters)
b = 1e-3  # Fixed plate separation (meters)
d = 0.5 * c / np.median(fn)  # ULA element spacing

# Angle range for estimation (in degrees)
N_phi = 41
phis = np.linspace(0, 40, N_phi)  # Updated grid for estimation to 0-30

# SNR levels (dB)
SNR_dB = np.linspace(-5, 15, 10)  # Updated SNR range for consistency
SNR = 10 ** (SNR_dB / 10)

# Monte Carlo simulation settings
num_MC = 1000

# Element count for ULA
ne = 5

# Preallocate error arrays
errors_LWA = np.zeros((num_MC, len(SNR_dB)))
errors_inner = np.zeros((num_MC, len(SNR_dB)))
errors_ULA = np.zeros((num_MC, len(SNR_dB)))
crb_LWA = np.zeros((num_MC, len(SNR_dB)))
crb_LWA_analytic = np.zeros((num_MC, len(SNR_dB)))


def lwa_signal(theta):
    G = np.array([np.sinc(((2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) -
                            2 * np.pi * f / c * np.cos(theta)) * L / 2))
                  for f in fn])
    return G / np.linalg.norm(G)


def analytic_dG_dtheta(f, theta, c, b, L):
    u = (2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) -
         2 * np.pi * f / c * np.cos(theta)) * L / 2
    if np.abs(u) < 1e-8:
        sinc_prime = 0.0
    else:
        sinc_prime = (u * np.cos(u) - np.sin(u)) / (u ** 2)
    du_dtheta = (np.pi * f * L / c) * np.sin(theta)
    return sinc_prime * du_dtheta


def analytic_crb(theta, fn, c, b, L, sigma2):
    deriv_sq_sum = 0.0
    for f in fn:
        deriv_sq_sum += analytic_dG_dtheta(f, theta, c, b, L) ** 2
    if deriv_sq_sum > 0:
        return sigma2 / (2 * deriv_sq_sum)
    else:
        return np.inf


delta = 1e-5

for mc in range(num_MC):
    user_angle = np.random.uniform(0, 30)
    user_angles = [user_angle]

    G = np.array([np.sinc(((2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) -
                            2 * np.pi * f / c * np.cos(np.deg2rad(user_angle))) * L / 2))
                  for f in fn])
    G_combined = G / np.linalg.norm(G)

    for idx_si, si in enumerate(SNR):
        current_sigma2 = 1 / si
        noise_LWA = np.sqrt(current_sigma2) * np.random.randn(N_f)
        received_signals = G_combined + noise_LWA

        x_noisy = np.zeros((ne, N_f), dtype=complex)
        for fi in range(N_f):
            wavelength = c / fn[fi]
            k = 2 * np.pi / wavelength
            steering_vector = np.exp(-1j * k * d * np.arange(ne) * np.sin(np.deg2rad(user_angle)))
            noise_vec = np.sqrt(current_sigma2 / 2) * (np.random.randn(ne) + 1j * np.random.randn(ne))
            x_noisy[:, fi] += steering_vector + noise_vec

        # --- LWA Grid Search ---
        best_l2_norm = np.inf
        estimated_angle_LWA = 0
        for phi in phis:
            est_sig = np.array([np.sinc(((2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) -
                                          2 * np.pi * f / c * np.cos(np.deg2rad(phi))) * L / 2))
                                for f in fn])
            residual = np.linalg.norm(est_sig - received_signals)
            if residual < best_l2_norm:
                best_l2_norm = residual
                estimated_angle_LWA = phi

        # --- Inner Product Method ---
        inner_product_values = []
        for phi in phis:
            modeled_signals = np.array([np.sinc(((2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) -
                                                  2 * np.pi * f / c * np.cos(np.deg2rad(phi))) * L / 2))
                                        for f in fn])
            modeled_signals_normalized = modeled_signals / np.linalg.norm(modeled_signals)
            ip_val = abs(np.real(np.vdot(received_signals, modeled_signals_normalized)))
            inner_product_values.append(ip_val)
        max_idx = np.argmax(inner_product_values)
        estimated_angle_inner = phis[max_idx]

        # --- ULA Beamforming Method ---
        angles = np.arange(0, 31, 1)  # Updated angle range 0-30
        responses = np.zeros(len(angles))
        for ai, angle in enumerate(angles):
            for fi in range(N_f):
                wavelength = c / fn[fi]
                k = 2 * np.pi / wavelength
                a = np.exp(-1j * k * d * np.arange(ne) * np.sin(np.deg2rad(angle)))
                responses[ai] += np.abs(np.dot(a.conj(), x_noisy[:, fi])) ** 2
        max_idx_ula = np.argmax(responses)
        estimated_angle_ULA = angles[max_idx_ula]

        errors_LWA[mc, idx_si] = permutation_invariant_rmse(user_angles, [estimated_angle_LWA])
        errors_inner[mc, idx_si] = permutation_invariant_rmse(user_angles, [estimated_angle_inner])
        errors_ULA[mc, idx_si] = permutation_invariant_rmse(user_angles, [estimated_angle_ULA])

        theta_fd = np.deg2rad(user_angle)
        s = lwa_signal(theta_fd)
        ds = (lwa_signal(theta_fd + delta) - lwa_signal(theta_fd - delta)) / (2 * delta)
        J11 = np.dot(ds, ds)
        if J11 > 0:
            crb_val_fd = np.sqrt(current_sigma2 / J11)
        else:
            crb_val_fd = np.inf
        crb_LWA[mc, idx_si] = crb_val_fd

        theta_an = np.deg2rad(user_angle)
        crb_val_an = np.sqrt(analytic_crb(theta_an, fn, c, b, L, current_sigma2))
        crb_LWA_analytic[mc, idx_si] = crb_val_an

# Average the errors over Monte Carlo simulations
total_MSE_LWA = np.mean(errors_LWA, axis=0)
total_MSE_inner = np.mean(errors_inner, axis=0)
total_MSE_ULA = np.mean(errors_ULA, axis=0)
total_CRB_LWA_an = np.mean(crb_LWA_analytic, axis=0)

# Plot the average RMSE vs SNR for each method with consistent style
plt.figure(figsize=(12, 6))
plt.plot(SNR_dB, total_MSE_LWA, marker='o', linestyle='-', color='b', linewidth=2, markersize=8,
         label='LWA Grid Search')
plt.plot(SNR_dB, total_MSE_inner, marker='s', linestyle='--', color='r', linewidth=2, markersize=8,
         label='LWA Beamforming')
plt.plot(SNR_dB, total_MSE_ULA, marker='x', linestyle='-.', color='g', linewidth=2, markersize=8,
         label='ULA Beamforming')
line_analytic, = plt.plot(SNR_dB, total_CRB_LWA_an, marker='^', linestyle='-', color='m', linewidth=2, markersize=8,
                          label='LWA CRB')
line_analytic.set_dashes([3, 1, 1, 1])
plt.xlabel('SNR (dB)', fontsize=16)
plt.ylabel('Average RMSE (radians)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.show()
