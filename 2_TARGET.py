import numpy as np
import matplotlib.pyplot as plt


def permutation_invariant_rmse(true_angles, est_angles):
    # Convert angles from degrees to radians
    true_angles = np.array(true_angles) * np.pi / 180
    est_angles = np.array(est_angles) * np.pi / 180
    # Compute error for direct pairing
    err1 = np.mean((true_angles - est_angles) ** 2)
    # Compute error for swapped pairing
    err2 = np.mean((true_angles - est_angles[::-1]) ** 2)
    return np.sqrt(min(err1, err2))


# Constants and Parameters
c = 3e8  # Speed of light (m/s)
N_f = 40  # Number of frequencies
fmin = 200e9  # Minimum frequency (Hz)
fmax = 800e9  # Maximum frequency (Hz)
fn = np.linspace(fmin, fmax, N_f)  # Frequency vector
L = 25e-3  # Fixed slit length (meters)
b = 1e-3  # Fixed plate separation (meters)
d = 0.5 * c / np.median(fn)  # ULA element spacing (half-wavelength at median frequency)

# Angle range for estimation (in degrees)
N_phi = 36
phis = np.linspace(0, 35, N_phi)  # Grid for estimation

# SNR levels
SNR_dB = np.linspace(-5, 15, 10)  # SNR range (dB)
SNR = 10 ** (SNR_dB / 10)  # Linear SNR values

# Monte Carlo simulation settings
num_MC = 1000  # Number of Monte Carlo simulations

# Element count for ULA
ne = 8

# Threshold distance between angles (degrees)
angle_threshold = 5

# Preallocate error arrays: rows = MC iterations, columns = SNR values
errors_LWA = np.zeros((num_MC, len(SNR_dB)))
errors_inner = np.zeros((num_MC, len(SNR_dB)))
errors_ULA = np.zeros((num_MC, len(SNR_dB)))
crb_LWA = np.zeros((num_MC, len(SNR_dB)))           # Numerical CRB for LWA
crb_LWA_analytic = np.zeros((num_MC, len(SNR_dB)))    # Analytic CRB for LWA


# Define LWA signal model function (theta in radians)
def lwa_signal(theta):
    # Compute the modeled response over frequencies
    G = np.array([np.sinc((2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) -
                           2 * np.pi * f / c * np.cos(theta)) * L / 2) for f in fn])
    return G / np.linalg.norm(G)


# Compute the analytic derivative of the normalized LWA signal with respect to theta
def lwa_signal_normalized_derivative(theta):
    # Unnormalized signal: use np.sinc(x) = sin(pi*x)/(pi*x)
    u = (2 * np.pi * fn / c * np.sqrt(1 - (c / (2 * b * fn)) ** 2) - 2 * np.pi * fn / c * np.cos(theta)) * L / 2
    G = np.sinc(u)
    normG = np.linalg.norm(G)

    # Derivative of np.sinc(u) with respect to u:
    # For u ≠ 0: d/du np.sinc(u) = [pi*u*cos(pi*u) - sin(pi*u)]/(pi*u**2)
    dsinc_du = np.zeros_like(u)
    nonzero = u != 0
    dsinc_du[nonzero] = (np.pi * u[nonzero] * np.cos(np.pi * u[nonzero]) - np.sin(np.pi * u[nonzero])) / (np.pi * u[nonzero] ** 2)
    dsinc_du[~nonzero] = 0.0  # derivative at 0 is 0

    # Derivative of u with respect to theta (only the cos(theta) term depends on theta)
    du_dtheta = 2 * np.pi * fn / c * np.sin(theta) * L / 2  # derivative of -cos(theta) gives sin(theta)

    dG = dsinc_du * du_dtheta
    # Now compute derivative of the normalized signal s = G/||G||
    d_normG = np.dot(G.conj(), dG) / normG
    ds = dG / normG - G * d_normG / (normG ** 2)
    return ds


# Small perturbation for finite differences (in radians)
delta = 1e-6

# Outer loop over Monte Carlo simulations
for mc in range(num_MC):
    # Randomize user angles (in [0,30]) ensuring at least 5° separation
    while True:
        rand_angles = np.sort(np.random.uniform(0, 30, 2))
        if (rand_angles[1] - rand_angles[0]) >= angle_threshold:
            break
    user_angles = rand_angles.tolist()

    # Precompute the "true" combined LWA signal (without noise)
    G_combined = np.zeros(N_f, dtype=complex)
    for angle in user_angles:
        # Use the same modeled response as in CRB but with degrees->radians conversion
        G = np.array([np.sinc((2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) -
                               2 * np.pi * f / c * np.cos(np.deg2rad(angle))) * L / 2) for f in fn])
        G_combined += G / np.linalg.norm(G)
    G_combined /= np.linalg.norm(G_combined)  # Normalize the combined channel

    # For each SNR, add noise to the signals and perform localization
    for idx_si, si in enumerate(SNR):
        current_sigma2 = 1 / si  # Noise variance for current SNR

        # LWA: Received signal = G_combined + noise (noise added per frequency)
        noise_LWA = np.sqrt(current_sigma2) * np.random.randn(N_f)
        received_signals = G_combined + noise_LWA

        # ULA: For each frequency, generate a noisy steering signal
        x_noisy = np.zeros((ne, N_f), dtype=complex)
        for fi in range(N_f):
            wavelength = c / fn[fi]
            k = 2 * np.pi / wavelength
            for angle in user_angles:
                steering_vector = np.exp(-1j * k * d * np.arange(ne) * np.sin(np.deg2rad(angle)))
                noise_vec = np.sqrt(current_sigma2 / 2) * (np.random.randn(ne) + 1j * np.random.randn(ne))
                x_noisy[:, fi] += steering_vector + noise_vec

        # --- LWA Grid Search ---
        best_l2_norm = np.inf
        estimated_angles_LWA = [0, 0]
        for phi1 in phis:
            for phi2 in phis:
                if phi2 - phi1 >= angle_threshold:
                    # Sum the modeled responses for the candidate pair
                    est_sig = np.zeros(N_f, dtype=complex)
                    for angle in [phi1, phi2]:
                        est_sig += np.array([np.sinc((2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) -
                                                      2 * np.pi * f / c * np.cos(np.deg2rad(angle))) * L / 2)
                                             for f in fn])
                    residual = np.linalg.norm(est_sig - received_signals)
                    if residual < best_l2_norm:
                        best_l2_norm = residual
                        estimated_angles_LWA = [phi1, phi2]

        # --- Inner Product Method ---
        inner_product_values = []
        for phi in phis:
            modeled_signals = np.array([np.sinc((2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) -
                                                 2 * np.pi * f / c * np.cos(np.deg2rad(phi))) * L / 2)
                                        for f in fn])
            modeled_signals_normalized = modeled_signals / np.linalg.norm(modeled_signals)
            ip_val = abs(np.real(np.vdot(received_signals, modeled_signals_normalized)))
            inner_product_values.append(ip_val)
        max_idx = np.argmax(inner_product_values)
        phi1_est = phis[max_idx]
        candidate_indices = [i for i, phi in enumerate(phis) if abs(phi - phi1_est) >= angle_threshold]
        if candidate_indices:
            candidate_ip_values = [inner_product_values[i] for i in candidate_indices]
            second_idx = candidate_indices[np.argmax(candidate_ip_values)]
            phi2_est = phis[second_idx]
        else:
            phi2_est = 0  # fallback
        estimated_angles_inner = [phi1_est, phi2_est]

        # --- ULA Beamforming Method ---
        angles = np.arange(0, 91, 1)
        responses = np.zeros(len(angles))
        for ai, angle in enumerate(angles):
            for fi in range(N_f):
                wavelength = c / fn[fi]
                k = 2 * np.pi / wavelength
                a = np.exp(-1j * k * d * np.arange(ne) * np.sin(np.deg2rad(angle)))
                responses[ai] += np.abs(np.dot(a.conj(), x_noisy[:, fi])) ** 2
        max_idx_ula = np.argmax(responses)
        phi1_ULA = angles[max_idx_ula]
        candidate_indices_ula = [i for i, angle in enumerate(angles) if abs(angle - phi1_ULA) >= angle_threshold]
        if candidate_indices_ula:
            candidate_vals = [responses[i] for i in candidate_indices_ula]
            second_idx_ula = candidate_indices_ula[np.argmax(candidate_vals)]
            phi2_ULA = angles[second_idx_ula]
        else:
            phi2_ULA = 0  # fallback
        estimated_angles_ULA = [phi1_ULA, phi2_ULA]

        # Compute permutation-invariant RMSE for each method
        errors_LWA[mc, idx_si] = permutation_invariant_rmse(user_angles, estimated_angles_LWA)
        errors_inner[mc, idx_si] = permutation_invariant_rmse(user_angles, estimated_angles_inner)
        errors_ULA[mc, idx_si] = permutation_invariant_rmse(user_angles, estimated_angles_ULA)

        # --- Numerical CRB Computation for LWA (using finite differences) ---
        theta1 = np.deg2rad(user_angles[0])
        theta2 = np.deg2rad(user_angles[1])
        ds1 = (lwa_signal(theta1 + delta) - lwa_signal(theta1 - delta)) / (2 * delta)
        ds2 = (lwa_signal(theta2 + delta) - lwa_signal(theta2 - delta)) / (2 * delta)
        J11 = np.dot(ds1, ds1)
        J22 = np.dot(ds2, ds2)
        J12 = np.dot(ds1, ds2)
        FIM = (1 / current_sigma2) * np.array([[J11, J12],
                                               [J12, J22]])
        crb_matrix = np.linalg.inv(FIM + 1e-12 * np.eye(2))
        crb_val = np.sqrt((crb_matrix[0, 0] + crb_matrix[1, 1]) / 2)
        crb_LWA[mc, idx_si] = crb_val

        # --- Analytic CRB Computation for LWA (using analytic derivatives) ---
        ds1_analytic = lwa_signal_normalized_derivative(theta1)
        ds2_analytic = lwa_signal_normalized_derivative(theta2)
        # Compute FIM elements (taking real parts to ensure numerical stability)
        J11_a = np.dot(ds1_analytic, ds1_analytic.conj()).real
        J22_a = np.dot(ds2_analytic, ds2_analytic.conj()).real
        J12_a = np.dot(ds1_analytic, ds2_analytic.conj()).real
        FIM_analytic = (1 / current_sigma2) * np.array([[J11_a, J12_a],
                                                        [J12_a, J22_a]])
        crb_matrix_analytic = np.linalg.inv(FIM_analytic + 1e-12 * np.eye(2))
        crb_val_analytic = np.sqrt((crb_matrix_analytic[0, 0] + crb_matrix_analytic[1, 1]) / 2)
        crb_LWA_analytic[mc, idx_si] = crb_val_analytic

        # For the first 2 MC simulations, print the estimated angles and CRB values for each SNR
        if mc < 2:
            print(f"SNR: {SNR_dB[idx_si]:.1f} dB, MC: {mc + 1}")
            print(f"  True User Angles: {user_angles}")
            print(f"  Estimated (LWA Grid Search): {estimated_angles_LWA}")
            print(f"  Estimated (Inner Product): {estimated_angles_inner}")
            print(f"  Estimated (ULA Beamforming): {estimated_angles_ULA}")
            print(f"  Analytic CRB (LWA): {crb_val_analytic:.4f} radians")

# Average the errors over Monte Carlo simulations (along axis 0)
total_MSE_LWA = np.mean(errors_LWA, axis=0)
total_MSE_inner = np.mean(errors_inner, axis=0)
total_MSE_ULA = np.mean(errors_ULA, axis=0)
total_CRB_LWA = np.mean(crb_LWA, axis=0)
total_CRB_LWA_analytic = np.mean(crb_LWA_analytic, axis=0)

# Plot the average RMSE vs SNR for each method, including both CRBs.
plt.figure(figsize=(12, 6))
plt.plot(SNR_dB, total_MSE_LWA, marker='o', linestyle='-', linewidth=2, markersize=8, label='LWA Grid Search ')
plt.plot(SNR_dB, total_MSE_inner, marker='s', linestyle='--', linewidth=2, markersize=8, label='LWA Beamforming ')
plt.plot(SNR_dB, total_MSE_ULA, marker='x', linestyle='-.', linewidth=2, markersize=8, label='ULA Beamforming ')
# For analytic CRB, we use a custom dash pattern to differentiate it further.
line_analytic, = plt.plot(SNR_dB, total_CRB_LWA_analytic, marker='^', linestyle='-', linewidth=2, markersize=8, label='LWA CRB')
line_analytic.set_dashes([3, 1, 1, 1])  # custom dash pattern

plt.xlabel('SNR (dB)', fontsize=16)
plt.ylabel('Average RMSE (radians)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.show()
