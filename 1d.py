import numpy as np
import matplotlib.pyplot as plt


def permutation_invariant_rmse(true_angles, est_angles):
    # For one target, this simply returns the absolute error.
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
fn = np.linspace(fmin, fmax, N_f)  # Frequency vector
L = 25e-3  # Fixed slit length (meters)
b = 1.1e-3  # Fixed plate separation (meters)
d = 0.5 * c / np.median(fn)  # ULA element spacing (half-wavelength at median frequency)

# Angle range for estimation (in degrees)
N_phi = 41
phis = np.linspace(0, 40, N_phi)  # Grid for estimation

# SNR levels
SNR_dB = np.linspace(-5, 30, 10)  # SNR range (dB)
SNR = 10 ** (SNR_dB / 10)  # Linear SNR values

# Monte Carlo simulation settings
num_MC = 1000  # Number of Monte Carlo simulations

# Element count for ULA
ne = 8

# Preallocate error arrays: rows = MC iterations, columns = SNR values
errors_LWA = np.zeros((num_MC, len(SNR_dB)))
errors_inner = np.zeros((num_MC, len(SNR_dB)))
errors_ULA = np.zeros((num_MC, len(SNR_dB)))
# CRB for LWA computed using finite differences
crb_LWA = np.zeros((num_MC, len(SNR_dB)))
# CRB for LWA computed analytically using the derivative with respect to theta
crb_LWA_analytic = np.zeros((num_MC, len(SNR_dB)))


# Define LWA signal model function (theta in radians)
def lwa_signal(theta):
    # Compute the modeled response over frequencies using sinc(x)= sin(x)/x.
    G = np.array([np.sinc(
        ((2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) - 2 * np.pi * f / c * np.cos(theta)) * L / 2))
                  for f in fn])
    return G / np.linalg.norm(G)


# Analytic derivative of G with respect to theta (in radians)
def analytic_dG_dtheta(f, theta, c, b, L):
    """
    Computes the derivative of the LWA response G(f,theta) with respect to theta.
    The signal model is:
      G(f,theta) = sinc(u(f,theta))
    with
      u(f,theta) = ( (2*pi*f/c*sqrt(1-(c/(2*b*f))**2) - 2*pi*f/c*cos(theta) ) * L/2 ).

    Using the chain rule:
      dG/dtheta = [ (u*cos(u)-sin(u))/u^2 ] * d(u)/dtheta,
    and since only the -2*pi*f/c*cos(theta) term in u depends on theta,
      d(u)/dtheta = (pi*f*L/c)*sin(theta).
    """
    u = (2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) - 2 * np.pi * f / c * np.cos(theta)) * L / 2
    if np.abs(u) < 1e-8:
        sinc_prime = 0.0
    else:
        sinc_prime = (u * np.cos(u) - np.sin(u)) / (u ** 2)
    du_dtheta = (np.pi * f * L / c) * np.sin(theta)
    return sinc_prime * du_dtheta


def analytic_crb(theta, fn, c, b, L, sigma2):
    """
    Computes the analytic CRB for the angle theta using the derivative of the LWA response.
    The Fisher information is approximated as:
      I(theta) = 2 * sum_{f in fn} (dG/dtheta)^2,
    so that the CRB (variance) is given by:
      CRB(theta) = sigma^2 / I(theta).
    The function returns the variance, and the square root is taken later.
    """
    deriv_sq_sum = 0.0
    for f in fn:
        deriv_sq_sum += analytic_dG_dtheta(f, theta, c, b, L) ** 2
    if deriv_sq_sum > 0:
        return sigma2 / (2 * deriv_sq_sum)
    else:
        return np.inf


# Small perturbation for finite differences (in radians)
delta = 1e-6

# Outer loop over Monte Carlo simulations
for mc in range(num_MC):
    # Generate one random target angle in [0,30] degrees
    user_angle = np.random.uniform(0, 30)
    user_angles = [user_angle]  # True angle as a list

    # Precompute the "true" LWA signal (without noise) for the target
    G = np.array([np.sinc(((2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2) -
                            2 * np.pi * f / c * np.cos(np.deg2rad(user_angle))) * L / 2))
                  for f in fn])
    G_combined = G / np.linalg.norm(G)

    # For each SNR, add noise to the signals and perform localization
    for idx_si, si in enumerate(SNR):
        current_sigma2 = 1 / si  # Noise variance for current SNR

        # LWA: Received signal = G_combined + noise (added per frequency)
        noise_LWA = np.sqrt(current_sigma2) * np.random.randn(N_f)
        received_signals = G_combined + noise_LWA

        # ULA: For each frequency, generate a noisy steering signal for the target
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
        angles = np.arange(0, 91, 1)
        responses = np.zeros(len(angles))
        for ai, angle in enumerate(angles):
            for fi in range(N_f):
                wavelength = c / fn[fi]
                k = 2 * np.pi / wavelength
                a = np.exp(-1j * k * d * np.arange(ne) * np.sin(np.deg2rad(angle)))
                responses[ai] += np.abs(np.dot(a.conj(), x_noisy[:, fi])) ** 2
        max_idx_ula = np.argmax(responses)
        estimated_angle_ULA = angles[max_idx_ula]

        # Compute permutation-invariant RMSE for each method
        errors_LWA[mc, idx_si] = permutation_invariant_rmse(user_angles, [estimated_angle_LWA])
        errors_inner[mc, idx_si] = permutation_invariant_rmse(user_angles, [estimated_angle_inner])
        errors_ULA[mc, idx_si] = permutation_invariant_rmse(user_angles, [estimated_angle_ULA])

        # --- CRB Computation for LWA (Finite Difference) ---
        theta_fd = np.deg2rad(user_angle)
        s = lwa_signal(theta_fd)
        ds = (lwa_signal(theta_fd + delta) - lwa_signal(theta_fd - delta)) / (2 * delta)
        J11 = np.dot(ds, ds)
        if J11 > 0:
            crb_val_fd = np.sqrt(current_sigma2 / J11)
        else:
            crb_val_fd = np.inf
        crb_LWA[mc, idx_si] = crb_val_fd

        # --- CRB Computation for LWA (Analytic) using derivative with respect to theta ---
        theta_an = np.deg2rad(user_angle)
        crb_val_an = np.sqrt(analytic_crb(theta_an, fn, c, b, L, current_sigma2))
        crb_LWA_analytic[mc, idx_si] = crb_val_an

        # For the first 2 MC simulations, print the estimated angle for each SNR
        if mc < 2:
            print(f"SNR: {SNR_dB[idx_si]:.1f} dB, MC: {mc + 1}")
            print(f"  True Target Angle (deg): {user_angle:.2f}")
            print(f"  Estimated (LWA Grid Search): {estimated_angle_LWA:.2f}")
            print(f"  Estimated (Inner Product): {estimated_angle_inner:.2f}")
            print(f"  Estimated (ULA Beamforming): {estimated_angle_ULA:.2f}")
            print(f"  CRB (Finite Diff.): {crb_val_fd:.4f} radians")
            print(f"  CRB (Analytic): {crb_val_an:.4f} radians")

# Average the errors over Monte Carlo simulations (along axis 0)
total_MSE_LWA = np.mean(errors_LWA, axis=0)
total_MSE_inner = np.mean(errors_inner, axis=0)
total_MSE_ULA = np.mean(errors_ULA, axis=0)
total_CRB_LWA_an = np.mean(crb_LWA_analytic, axis=0)

# Plot the average RMSE vs SNR for each method in b-w format
plt.figure(figsize=(12, 6))
plt.plot(SNR_dB, total_MSE_LWA, 'o-', color='b', linewidth=2, label='LWA Grid Search')
plt.plot(SNR_dB, total_MSE_inner, 's--', color='r', linewidth=2, label='LWA Beamforming')
plt.plot(SNR_dB, total_MSE_ULA, 'x-.', color='g', linewidth=2, label='ULA Beamforming')
plt.plot(SNR_dB, total_CRB_LWA_an, 'd:', color='m', linewidth=2, label='CRB (Analytic)')
plt.xlabel('SNR (dB)')
plt.ylabel('Average RMSE (radians)')
plt.grid(True)
plt.legend()
plt.show()
