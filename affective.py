import numpy as np
import matplotlib.pyplot as plt


def lwa_channel_response(fn, angle, c, b, L):
    """
    Computes the LWA channel response (sinc-based) for a given angle.
    """
    return np.array([
        np.sinc(
            (2 * np.pi * f / c * np.sqrt(1 - (c / (2 * b * f)) ** 2)
             - 2 * np.pi * f / c * np.cos(np.deg2rad(angle)))
            * L / 2
        )
        for f in fn
    ])


def assign_angles(candidate1, candidate2, theta1_true, theta2_true):
    """
    Given two candidate angles and the true angles, compute the squared error
    for both possible assignments and return the assignment that minimizes the total error.
    """
    error_perm1 = (np.deg2rad(candidate1) - np.deg2rad(theta1_true)) ** 2 \
                  + (np.deg2rad(candidate2) - np.deg2rad(theta2_true)) ** 2
    error_perm2 = (np.deg2rad(candidate2) - np.deg2rad(theta1_true)) ** 2 \
                  + (np.deg2rad(candidate1) - np.deg2rad(theta2_true)) ** 2

    if error_perm1 <= error_perm2:
        return candidate1, candidate2
    else:
        return candidate2, candidate1


# -----------------------------
# Common Simulation Parameters
# -----------------------------
c = 3e8  # Speed of light (m/s)
fmin = 200e9  # Minimum frequency (Hz)
fmax = 800e9  # Maximum frequency (Hz)
N_f = 40  # Number of frequency bins (subcarriers)
fn = np.linspace(fmin, fmax, N_f)
L = 25e-3  # Slit length (m)
b = 1e-3  # Plate separation (m)
phases = np.zeros(N_f)
x = np.exp(1j * phases)

# -----------------------------
# One-Angle Estimation Simulation
# -----------------------------
candidate_angles = np.linspace(0, 90, 90)  # Candidate angles (0° to 90°)
theta_values = np.arange(5, 86, 5)  # True angle values from 5° to 85°
num_mc = 500  # Number of Monte Carlo iterations per angle

rmse_results_one = []
np.random.seed(0)  # For reproducibility

for theta in theta_values:
    errors = []
    for mc in range(num_mc):
        # Generate channel response for the true angle
        G = lwa_channel_response(fn, theta, c, b, L)
        Y = x * G
        Y_norm = Y / np.linalg.norm(Y)

        # -----------------------------
        # Add Noise after Normalization (SNR = 10 dB)
        # -----------------------------
        snr_db = 10
        snr_linear = 10 ** (snr_db / 10)
        noise_std = np.sqrt(1 / (snr_linear * 2))
        noise = noise_std * np.random.randn(N_f)
        Y_noisy = Y_norm + noise
        Y_noisy_norm = Y_noisy / np.linalg.norm(Y_noisy)

        # Grid Search for Angle Estimation
        metrics = []
        for candidate in candidate_angles:
            G_candidate = lwa_channel_response(fn, candidate, c, b, L)
            Y_candidate = x * G_candidate
            Y_candidate_norm = Y_candidate / np.linalg.norm(Y_candidate)
            metric_value = np.abs(np.vdot(Y_candidate_norm, Y_noisy_norm))
            metrics.append(metric_value)
        metrics = np.array(metrics)
        best_candidate = candidate_angles[np.argmax(metrics)]

        # Compute squared error (in radians)
        error = (np.deg2rad(best_candidate) - np.deg2rad(theta)) ** 2
        errors.append(error)

    rmse = np.sqrt(np.mean(errors))
    rmse_results_one.append(rmse)
    print(f"One-Angle: True Angle {theta}°, RMSE: {rmse:.4f} radians")

# -----------------------------
# Two-Angle Estimation Simulation
# -----------------------------
candidate_theta2 = np.linspace(0, 90, 180)  # Candidate angles with finer resolution
theta1_values = np.arange(5, 86, 5)  # True theta1 values from 5° to 85°
num_mc = 500  # Number of Monte Carlo iterations per theta1

rmse_results_two = []
np.random.seed(0)  # Reset seed for reproducibility

for theta1 in theta1_values:
    errors_theta1 = []
    errors_theta2 = []
    for mc in range(num_mc):
        # Ensure the true theta2 is at least 5° from theta1
        lower_bound = max(0, theta1 - 10)
        upper_bound = min(90, theta1 + 10)
        while True:
            theta2_candidate = np.random.uniform(lower_bound, upper_bound)
            if np.abs(theta2_candidate - theta1) >= 5:
                theta2_true = theta2_candidate
                break

        # Generate channel responses for theta1 and theta2
        G1 = lwa_channel_response(fn, theta1, c, b, L)
        G2 = lwa_channel_response(fn, theta2_true, c, b, L)

        # Form the received signal (both transmitters send the same pilot)
        Y = x * (G1 + G2)
        Y_norm = Y / np.linalg.norm(Y)

        # -----------------------------
        # Add Noise after Normalization (SNR = 1000 dB)
        # -----------------------------
        snr_db = 1000
        snr_linear = 10 ** (snr_db / 10)
        noise_std = np.sqrt(1 / (snr_linear * 2))
        noise = noise_std * np.random.randn(N_f)
        Y_noisy = Y_norm + noise
        Y_noisy_norm = Y_noisy / np.linalg.norm(Y_noisy)

        # Grid Search for Two-Peak Estimation
        metrics = []
        for theta_candidate in candidate_theta2:
            G_candidate = lwa_channel_response(fn, theta_candidate, c, b, L)
            Y_candidate = x * G_candidate
            Y_candidate_norm = Y_candidate / np.linalg.norm(Y_candidate)
            metric_value = np.abs(np.vdot(Y_candidate_norm, Y_noisy_norm))
            metrics.append(metric_value)
        metrics = np.array(metrics)

        # Find the two highest peaks with at least 5° separation
        sorted_indices_desc = np.argsort(metrics)[::-1]
        candidate1 = candidate_theta2[sorted_indices_desc[0]]
        candidate2 = None
        for idx in sorted_indices_desc[1:]:
            if np.abs(candidate1 - candidate_theta2[idx]) >= 5:
                candidate2 = candidate_theta2[idx]
                break
        if candidate2 is None:
            candidate2 = candidate_theta2[sorted_indices_desc[1]]

        # Assign estimated angles to the two transmitters
        theta1_est, theta2_est = assign_angles(candidate1, candidate2, theta1, theta2_true)

        # Final check: ensure the estimated angles are at least 5° apart.
        if np.abs(theta1_est - theta2_est) < 5:
            continue

        error_theta1 = (np.deg2rad(theta1_est) - np.deg2rad(theta1)) ** 2
        error_theta2 = (np.deg2rad(theta2_est) - np.deg2rad(theta2_true)) ** 2
        errors_theta1.append(error_theta1)
        errors_theta2.append(error_theta2)

    total_rmse = np.sqrt((np.mean(errors_theta1) + np.mean(errors_theta2)) / 2)
    rmse_results_two.append(total_rmse)
    print(f"Two-Angle: Theta1 {theta1}°, Total RMSE: {total_rmse:.4f} radians")

# -----------------------------
# Combined Plot
# -----------------------------
plt.figure(figsize=(10, 6))
# One-Angle RMSE: solid line with circle markers
plt.plot(theta_values, rmse_results_one, marker='o', linestyle='-', linewidth=2, color='black', label='One-Angle RMSE')
# Two-Angle Total RMSE: dashed line with square markers
plt.plot(theta1_values, rmse_results_two, marker='s', linestyle='--', linewidth=2, color='black', label='Two-Angle Total RMSE')
plt.xlabel('Angle (degrees)')
plt.ylabel('RMSE (radians)')
plt.title('Comparison of RMSE for One-Angle and Two-Angle Estimations')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

