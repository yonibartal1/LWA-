# LWA Channel Response and Angle Estimation Simulation

This repository contains Python code to simulate and analyze the performance of Leaky-Wave Antenna (LWA) channel responses for angle estimation. The simulations include both one-angle and two-angle estimation scenarios using various methods, such as grid search, LWA Beamforming, and ULA beamforming. In addition, the code computes performance metrics like RMSE and the Cramér-Rao Bound (CRB) using both finite difference and analytic derivative approaches.

## Overview

The repository covers:
- **LWA Channel Response Simulation:** Computes a sinc-based channel response over a range of frequencies.
- **One-Angle Estimation:** Uses grid search to estimate a single target angle and evaluates its RMSE.
- **Two-Angle Estimation:** Simultaneously estimates two target angles with a permutation-invariant error metric.
- **CRB Analysis:** Compares numerical CRB (using finite differences) and analytic CRB (using derivative calculations) for LWA.
- **ULA Beamforming Comparison:** Evaluates a conventional Uniform Linear Array (ULA) beamforming method for angle estimation.

## Features

- **Sinc-Based Modeling:** Models the LWA channel response using a sinc function.
- **Multiple Estimation Methods:** Implements grid search, inner product matching, and ULA beamforming.
- **Monte Carlo Simulations:** Performs multiple iterations to statistically assess estimation accuracy.
- **Performance Metrics:** Calculates RMSE and CRB to benchmark different methods.
- **Visualization:** Generates plots to compare estimation accuracy across various SNR levels and angle configurations.

## Requirements

- Python 3.12
- NumPy
- Matplotlib

Install the necessary libraries with:

```bash
pip install numpy matplotlib
