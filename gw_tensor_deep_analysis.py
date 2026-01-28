#!/usr/bin/env python3
"""
Deep Tensor Analysis for Gravitational Wave Data

PURPOSE: More sensitive, properly normalized tensor analysis of GW data.
This addresses the amplitude scaling differences between EEG and GW signals.

Key insight: The tensor's RESPONSE PATTERN matters more than absolute symmetry.
We look at:
1. Normalized peak detection
2. Temporal evolution of tensor response
3. Gradient structure
4. Spectral transformation
5. Comparison of tensor response across signal types

Author: TEAPOT Cross-Domain Analysis
Date: January 2026
"""

import numpy as np
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AdaptiveTensorOperator:
    """
    Tensor operator with adaptive thresholding for cross-domain analysis.
    """

    def __init__(self, coefficients=None):
        if coefficients is None:
            self.coefficients = np.array([0.1, 0.2, 0.15, 0.25, 0.3,
                                         0.2, 0.15, 0.1, 0.05, 0.1])
        else:
            self.coefficients = np.array(coefficients)

    def apply(self, signal_window):
        """Apply tensor transformation."""
        transformed = np.convolve(signal_window, self.coefficients, mode='valid')
        return transformed

    def compute_symmetry_adaptive(self, sig):
        """
        Compute symmetry with adaptive thresholding based on signal statistics.
        """
        # Adaptive threshold based on signal distribution
        threshold = np.std(sig) * 0.5

        peaks = self._detect_peaks_adaptive(sig, threshold)
        troughs = self._detect_peaks_adaptive(-sig, threshold)

        if len(peaks) == 0 or len(troughs) == 0:
            return 1.0, {'peaks': 0, 'troughs': 0}

        ratio = min(len(peaks), len(troughs)) / max(len(peaks), len(troughs))
        symmetry = abs(1.0 - ratio)

        return symmetry, {'peaks': len(peaks), 'troughs': len(troughs)}

    def _detect_peaks_adaptive(self, sig, threshold):
        """Peak detection with adaptive threshold."""
        peaks = []
        for i in range(1, len(sig) - 1):
            if sig[i] > sig[i-1] and sig[i] > sig[i+1]:
                if sig[i] > threshold:
                    peaks.append(i)
        return peaks

    def compute_tensor_gradient(self, sig):
        """Compute gradient of tensor-transformed signal."""
        transformed = self.apply(sig)
        gradient = np.gradient(transformed)
        return gradient

    def compute_tensor_curvature(self, sig):
        """Compute second derivative (curvature) of tensor output."""
        transformed = self.apply(sig)
        curvature = np.gradient(np.gradient(transformed))
        return curvature


def normalize_signal(sig):
    """Normalize signal to zero mean, unit variance."""
    sig = np.array(sig, dtype=float)
    mean = np.mean(sig)
    std = np.std(sig)
    if std > 0:
        return (sig - mean) / std
    return sig - mean


def generate_realistic_gw_event(event_params, sample_rate=4096, duration=32):
    """
    Generate realistic gravitational wave signal using physical parameters.
    """
    t = np.linspace(-duration/2, duration/2, int(sample_rate * duration))

    # Chirp mass calculation
    M1 = event_params['mass1']
    M2 = event_params['mass2']
    Mc = (M1 * M2)**(3/5) / (M1 + M2)**(1/5)

    # Frequency evolution
    tau = np.abs(t) + 0.01
    f_peak = event_params['peak_frequency']
    f = f_peak * (tau / 0.05)**(-3/8)
    f = np.clip(f, 20, f_peak * 1.2)

    # Amplitude evolution (increases toward merger)
    amplitude = (f / f_peak)**(2/3)

    # Merger cutoff
    merger_idx = len(t) // 2
    amplitude[merger_idx:] *= np.exp(-10 * (t[merger_idx:])**2)

    # Phase accumulation
    phase = 2 * np.pi * np.cumsum(f) / sample_rate

    # Waveform (plus polarization)
    h_plus = amplitude * np.cos(phase)

    # Add realistic noise (colored Gaussian)
    noise_level = 0.3  # Signal-to-noise ratio
    noise = np.random.randn(len(t)) * noise_level

    strain = h_plus + noise

    return strain, t, sample_rate


def analyze_tensor_response(operator, signal_data, signal_name):
    """
    Comprehensive tensor response analysis.
    """
    # Normalize the signal for consistent tensor response
    normalized = normalize_signal(signal_data)

    # Apply tensor
    transformed = operator.apply(normalized)
    transformed_norm = normalize_signal(transformed)

    # Core metrics with adaptive thresholding
    symmetry, counts = operator.compute_symmetry_adaptive(transformed_norm)

    # Gradient analysis
    gradient = operator.compute_tensor_gradient(normalized)
    curvature = operator.compute_tensor_curvature(normalized)

    # Temporal structure (windowed analysis)
    window_size = len(transformed_norm) // 20  # 20 windows
    window_symmetries = []
    window_energies = []

    for i in range(20):
        start = i * window_size
        end = start + window_size
        if end <= len(transformed_norm):
            window = transformed_norm[start:end]
            w_sym, _ = operator.compute_symmetry_adaptive(window)
            window_symmetries.append(w_sym)
            window_energies.append(np.sum(window**2))

    # Spectral analysis of tensor output
    fft_transform = np.fft.rfft(transformed_norm)
    power_spectrum = np.abs(fft_transform)**2
    freqs = np.fft.rfftfreq(len(transformed_norm))

    # Find dominant frequencies
    peak_freq_idx = np.argsort(power_spectrum)[-5:]
    dominant_freqs = freqs[peak_freq_idx]

    # Higher-order statistics of tensor output
    skewness = stats.skew(transformed_norm)
    kurtosis = stats.kurtosis(transformed_norm)

    # Zero-crossing rate (related to frequency content)
    zero_crossings = np.sum(np.diff(np.sign(transformed_norm)) != 0)
    zcr = zero_crossings / len(transformed_norm)

    # Gradient statistics
    grad_mean = np.mean(np.abs(gradient))
    grad_std = np.std(gradient)
    grad_max = np.max(np.abs(gradient))

    # Curvature statistics
    curv_mean = np.mean(np.abs(curvature))
    curv_std = np.std(curvature)

    return {
        'signal_name': signal_name,
        'symmetry_delta': float(symmetry),
        'peak_count': counts['peaks'],
        'trough_count': counts['troughs'],
        'peak_trough_ratio': counts['peaks'] / counts['troughs'] if counts['troughs'] > 0 else float('inf'),
        'window_symmetries': window_symmetries,
        'symmetry_variance': float(np.var(window_symmetries)),
        'symmetry_evolution': float(np.corrcoef(range(len(window_symmetries)), window_symmetries)[0, 1]) if len(window_symmetries) > 1 else 0,
        'energy_evolution': window_energies,
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'zero_crossing_rate': float(zcr),
        'gradient_mean': float(grad_mean),
        'gradient_std': float(grad_std),
        'gradient_max': float(grad_max),
        'curvature_mean': float(curv_mean),
        'curvature_std': float(curv_std),
        'dominant_frequencies': dominant_freqs.tolist(),
        'total_spectral_power': float(np.sum(power_spectrum))
    }


def run_deep_analysis():
    """Run comprehensive tensor analysis on GW data."""

    print("=" * 80)
    print("DEEP TENSOR ANALYSIS - GRAVITATIONAL WAVE DATA")
    print("=" * 80)
    print()
    print("This analysis uses normalized signals and adaptive thresholding")
    print("to observe tensor response patterns across different signal types.")
    print()

    operator = AdaptiveTensorOperator()
    results = []

    # Define GW events with real physical parameters
    events = {
        'GW150914': {
            'description': 'First detection - Binary Black Hole',
            'mass1': 36, 'mass2': 29,
            'distance_mpc': 410,
            'peak_frequency': 150
        },
        'GW170817': {
            'description': 'Binary Neutron Star Merger',
            'mass1': 1.46, 'mass2': 1.27,
            'distance_mpc': 40,
            'peak_frequency': 400
        },
        'GW190521': {
            'description': 'Intermediate-Mass Black Hole',
            'mass1': 85, 'mass2': 66,
            'distance_mpc': 5300,
            'peak_frequency': 60
        }
    }

    print("-" * 80)
    print("GRAVITATIONAL WAVE EVENTS (Realistic Waveforms)")
    print("-" * 80)

    for event_name, params in events.items():
        print(f"\n>>> {event_name}: {params['description']}")
        strain, t, sr = generate_realistic_gw_event(params)
        result = analyze_tensor_response(operator, strain, event_name)
        results.append(result)

        print(f"    Symmetry Δ: {result['symmetry_delta']:.6f}")
        print(f"    Peak/Trough: {result['peak_count']}/{result['trough_count']} (ratio: {result['peak_trough_ratio']:.4f})")
        print(f"    Skewness: {result['skewness']:.4f}")
        print(f"    Kurtosis: {result['kurtosis']:.4f}")
        print(f"    Zero-crossing rate: {result['zero_crossing_rate']:.6f}")
        print(f"    Gradient magnitude: {result['gradient_mean']:.6f} ± {result['gradient_std']:.6f}")

    # Control signals
    print("\n" + "-" * 80)
    print("CONTROL SIGNALS")
    print("-" * 80)

    n_samples = 4096 * 32  # Same duration as GW signals
    t_ctrl = np.linspace(0, 32, n_samples)

    controls = {
        'gaussian_noise': np.random.randn(n_samples),
        'pink_noise': np.cumsum(np.random.randn(n_samples)) / 100,
        'sinusoid_100Hz': np.sin(2*np.pi*100*t_ctrl),
        'chirp_50_200Hz': np.sin(2*np.pi*np.cumsum(50 + 150*t_ctrl/32)/4096),
        'step_function': np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)]),
        'damped_oscillation': np.exp(-t_ctrl/10) * np.sin(2*np.pi*50*t_ctrl)
    }

    for ctrl_name, ctrl_signal in controls.items():
        print(f"\n>>> Control: {ctrl_name}")
        result = analyze_tensor_response(operator, ctrl_signal, f"CTRL_{ctrl_name}")
        results.append(result)

        print(f"    Symmetry Δ: {result['symmetry_delta']:.6f}")
        print(f"    Peak/Trough: {result['peak_count']}/{result['trough_count']} (ratio: {result['peak_trough_ratio']:.4f})")
        print(f"    Skewness: {result['skewness']:.4f}")
        print(f"    Kurtosis: {result['kurtosis']:.4f}")
        print(f"    Zero-crossing rate: {result['zero_crossing_rate']:.6f}")

    # Comparative analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE TENSOR RESPONSE ANALYSIS")
    print("=" * 80)

    print("\n1. SYMMETRY COMPARISON:")
    sorted_by_sym = sorted(results, key=lambda x: x['symmetry_delta'])
    for r in sorted_by_sym:
        print(f"   {r['signal_name']:30s}: Δ = {r['symmetry_delta']:.6f}")

    print("\n2. PEAK-TROUGH BALANCE:")
    sorted_by_ratio = sorted(results, key=lambda x: abs(1 - x['peak_trough_ratio']))
    for r in sorted_by_ratio:
        print(f"   {r['signal_name']:30s}: ratio = {r['peak_trough_ratio']:.4f}")

    print("\n3. TENSOR GRADIENT RESPONSE:")
    sorted_by_grad = sorted(results, key=lambda x: x['gradient_mean'], reverse=True)
    for r in sorted_by_grad:
        print(f"   {r['signal_name']:30s}: grad = {r['gradient_mean']:.6f}")

    print("\n4. STATISTICAL MOMENTS:")
    print(f"   {'Signal':<30s} {'Skewness':>12s} {'Kurtosis':>12s}")
    for r in results:
        print(f"   {r['signal_name']:<30s} {r['skewness']:>12.4f} {r['kurtosis']:>12.4f}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS (Unbiased)")
    print("=" * 80)

    # Identify if GW events cluster differently from controls
    gw_results = [r for r in results if not r['signal_name'].startswith('CTRL_')]
    ctrl_results = [r for r in results if r['signal_name'].startswith('CTRL_')]

    gw_symmetries = [r['symmetry_delta'] for r in gw_results]
    ctrl_symmetries = [r['symmetry_delta'] for r in ctrl_results]

    gw_gradients = [r['gradient_mean'] for r in gw_results]
    ctrl_gradients = [r['gradient_mean'] for r in ctrl_results]

    gw_zcr = [r['zero_crossing_rate'] for r in gw_results]
    ctrl_zcr = [r['zero_crossing_rate'] for r in ctrl_results]

    print(f"\n  GW Events Average Symmetry: {np.mean(gw_symmetries):.6f} ± {np.std(gw_symmetries):.6f}")
    print(f"  Controls Average Symmetry:  {np.mean(ctrl_symmetries):.6f} ± {np.std(ctrl_symmetries):.6f}")

    print(f"\n  GW Events Average Gradient: {np.mean(gw_gradients):.6f} ± {np.std(gw_gradients):.6f}")
    print(f"  Controls Average Gradient:  {np.mean(ctrl_gradients):.6f} ± {np.std(ctrl_gradients):.6f}")

    print(f"\n  GW Events Average ZCR:      {np.mean(gw_zcr):.6f} ± {np.std(gw_zcr):.6f}")
    print(f"  Controls Average ZCR:       {np.mean(ctrl_zcr):.6f} ± {np.std(ctrl_zcr):.6f}")

    # Statistical test
    if len(gw_symmetries) >= 2 and len(ctrl_symmetries) >= 2:
        t_stat, p_value = stats.ttest_ind(gw_symmetries, ctrl_symmetries)
        print(f"\n  Symmetry Difference Test: t={t_stat:.4f}, p={p_value:.4f}")

        t_stat_g, p_value_g = stats.ttest_ind(gw_gradients, ctrl_gradients)
        print(f"  Gradient Difference Test: t={t_stat_g:.4f}, p={p_value_g:.4f}")

    # Save results
    results_dir = '/home/user/TEAPOT/results/gravitational_wave_analysis'
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, 'deep_tensor_analysis.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'methodology': 'normalized_adaptive_tensor_analysis',
            'results': results,
            'summary': {
                'gw_mean_symmetry': float(np.mean(gw_symmetries)),
                'ctrl_mean_symmetry': float(np.mean(ctrl_symmetries)),
                'gw_mean_gradient': float(np.mean(gw_gradients)),
                'ctrl_mean_gradient': float(np.mean(ctrl_gradients))
            }
        }, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_deep_analysis()
