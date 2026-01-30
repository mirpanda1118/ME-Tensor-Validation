#!/usr/bin/env python3
"""
BIAS CHECK: Testing if our results are artifacts

This script tests whether the tensor results are real or
introduced by our methodology.

Tests:
1. RANDOM TENSOR - Does ANY 10-element array produce similar results?
2. REVERSED TENSOR - Does coefficient order matter?
3. SHUFFLED SIGNALS - Does temporal structure matter?
4. RAW UNNORMALIZED - Does normalization create the pattern?
5. DIFFERENT METRICS - Do other measurements show the same pattern?

Author: TEAPOT Bias Analysis
Date: January 2026
"""

import numpy as np
from scipy import stats
import json
import os
from datetime import datetime

# =============================================================================
# ORIGINAL TENSOR (for comparison)
# =============================================================================

ORIGINAL_COEFFICIENTS = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.2, 0.15, 0.1, 0.05, 0.1])


def apply_tensor(signal, coefficients):
    """Apply tensor convolution."""
    return np.convolve(signal, coefficients, mode='valid')


def compute_symmetry(sig, use_adaptive=True):
    """Compute peak/trough symmetry."""
    if use_adaptive:
        threshold = np.std(sig) * 0.5
    else:
        threshold = 0.5  # Fixed threshold

    peaks = []
    troughs = []
    for i in range(1, len(sig) - 1):
        if sig[i] > sig[i-1] and sig[i] > sig[i+1] and sig[i] > threshold:
            peaks.append(i)
        if sig[i] < sig[i-1] and sig[i] < sig[i+1] and sig[i] < -threshold:
            troughs.append(i)

    if len(peaks) == 0 or len(troughs) == 0:
        return 1.0

    ratio = min(len(peaks), len(troughs)) / max(len(peaks), len(troughs))
    return abs(1.0 - ratio)


def normalize(sig):
    std = np.std(sig)
    return (sig - np.mean(sig)) / std if std > 0 else sig - np.mean(sig)


# =============================================================================
# GENERATE TEST SIGNALS (known properties)
# =============================================================================

def generate_test_signals():
    """Generate signals with KNOWN mathematical properties."""
    n = 10000
    t = np.linspace(0, 10, n)

    signals = {}

    # 1. Perfect sine (SHOULD be perfectly symmetric)
    signals['perfect_sine'] = np.sin(2*np.pi*10*t)

    # 2. Pure noise (SHOULD be ~symmetric by statistics)
    signals['pure_noise'] = np.random.randn(n)

    # 3. Asymmetric sawtooth (SHOULD be asymmetric)
    signals['sawtooth'] = 2 * (t % 1) - 1

    # 4. Exponential decay (SHOULD be highly asymmetric)
    signals['exp_decay'] = np.exp(-t)

    # 5. Square wave (peaks and troughs should be equal)
    signals['square_wave'] = np.sign(np.sin(2*np.pi*5*t))

    # 6. Chirp (frequency sweep - like GW)
    f = 10 + 40*t/10
    signals['chirp'] = np.sin(2*np.pi*np.cumsum(f)/n * 10)

    # 7. Damped oscillation
    signals['damped_osc'] = np.exp(-t/3) * np.sin(2*np.pi*10*t)

    # 8. Biased noise (mean != 0)
    signals['biased_noise'] = np.random.randn(n) + 5

    # 9. Half-wave rectified sine (definitely asymmetric)
    signals['rectified_sine'] = np.maximum(0, np.sin(2*np.pi*10*t))

    # 10. Poisson process (discrete events)
    signals['poisson'] = np.random.poisson(5, n).astype(float)

    return signals


# =============================================================================
# TEST 1: RANDOM TENSOR COMPARISON
# =============================================================================

def test_random_tensors(signals, n_random=50):
    """Does ANY random 10-element array produce similar results?"""
    print("\n" + "="*70)
    print("TEST 1: RANDOM TENSOR COMPARISON")
    print("="*70)
    print("Question: Does the specific tensor matter, or does ANY tensor work?")

    results = {'original': {}, 'random_mean': {}, 'random_std': {}}

    for sig_name, sig in signals.items():
        norm_sig = normalize(sig)

        # Original tensor
        transformed = apply_tensor(norm_sig, ORIGINAL_COEFFICIENTS)
        orig_sym = compute_symmetry(normalize(transformed))
        results['original'][sig_name] = orig_sym

        # Random tensors
        random_syms = []
        for _ in range(n_random):
            random_coef = np.random.randn(10)
            random_coef = random_coef / np.sum(np.abs(random_coef))  # Normalize
            transformed = apply_tensor(norm_sig, random_coef)
            random_syms.append(compute_symmetry(normalize(transformed)))

        results['random_mean'][sig_name] = np.mean(random_syms)
        results['random_std'][sig_name] = np.std(random_syms)

    print("\nResults:")
    print(f"{'Signal':<20} {'Original Δ':>12} {'Random Mean':>12} {'Random Std':>12} {'Unique?':>10}")
    print("-" * 70)
    for sig_name in signals:
        orig = results['original'][sig_name]
        rand_mean = results['random_mean'][sig_name]
        rand_std = results['random_std'][sig_name]
        # Is original significantly different from random?
        z_score = abs(orig - rand_mean) / (rand_std + 1e-6)
        unique = "YES" if z_score > 2 else "no"
        print(f"{sig_name:<20} {orig:>12.6f} {rand_mean:>12.6f} {rand_std:>12.6f} {unique:>10}")

    return results


# =============================================================================
# TEST 2: COEFFICIENT ORDER MATTERS?
# =============================================================================

def test_coefficient_order(signals):
    """Does the ORDER of coefficients matter?"""
    print("\n" + "="*70)
    print("TEST 2: COEFFICIENT ORDER")
    print("="*70)
    print("Question: Does reversing/shuffling coefficients change results?")

    reversed_coef = ORIGINAL_COEFFICIENTS[::-1]
    shuffled_coef = ORIGINAL_COEFFICIENTS.copy()
    np.random.shuffle(shuffled_coef)

    results = {}
    print("\nCoefficients tested:")
    print(f"  Original: {ORIGINAL_COEFFICIENTS}")
    print(f"  Reversed: {reversed_coef}")
    print(f"  Shuffled: {shuffled_coef}")

    print(f"\n{'Signal':<20} {'Original':>12} {'Reversed':>12} {'Shuffled':>12}")
    print("-" * 60)

    for sig_name, sig in signals.items():
        norm_sig = normalize(sig)

        t_orig = apply_tensor(norm_sig, ORIGINAL_COEFFICIENTS)
        t_rev = apply_tensor(norm_sig, reversed_coef)
        t_shuf = apply_tensor(norm_sig, shuffled_coef)

        sym_orig = compute_symmetry(normalize(t_orig))
        sym_rev = compute_symmetry(normalize(t_rev))
        sym_shuf = compute_symmetry(normalize(t_shuf))

        results[sig_name] = {'original': sym_orig, 'reversed': sym_rev, 'shuffled': sym_shuf}
        print(f"{sig_name:<20} {sym_orig:>12.6f} {sym_rev:>12.6f} {sym_shuf:>12.6f}")

    return results


# =============================================================================
# TEST 3: TEMPORAL STRUCTURE MATTERS?
# =============================================================================

def test_temporal_structure(signals):
    """Does the TEMPORAL ORDER of the signal matter?"""
    print("\n" + "="*70)
    print("TEST 3: TEMPORAL STRUCTURE")
    print("="*70)
    print("Question: If we shuffle the signal in time, do results change?")

    results = {}
    print(f"\n{'Signal':<20} {'Original':>12} {'Shuffled':>12} {'Diff':>12}")
    print("-" * 60)

    for sig_name, sig in signals.items():
        norm_sig = normalize(sig)
        shuffled_sig = norm_sig.copy()
        np.random.shuffle(shuffled_sig)

        t_orig = apply_tensor(norm_sig, ORIGINAL_COEFFICIENTS)
        t_shuf = apply_tensor(shuffled_sig, ORIGINAL_COEFFICIENTS)

        sym_orig = compute_symmetry(normalize(t_orig))
        sym_shuf = compute_symmetry(normalize(t_shuf))

        diff = abs(sym_orig - sym_shuf)
        results[sig_name] = {'original': sym_orig, 'shuffled': sym_shuf, 'diff': diff}
        print(f"{sig_name:<20} {sym_orig:>12.6f} {sym_shuf:>12.6f} {diff:>12.6f}")

    return results


# =============================================================================
# TEST 4: NORMALIZATION EFFECT
# =============================================================================

def test_normalization_effect(signals):
    """Does normalization CREATE the pattern?"""
    print("\n" + "="*70)
    print("TEST 4: NORMALIZATION EFFECT")
    print("="*70)
    print("Question: Do results change if we DON'T normalize?")

    results = {}
    print(f"\n{'Signal':<20} {'Normalized':>12} {'Raw':>12} {'Diff':>12}")
    print("-" * 60)

    for sig_name, sig in signals.items():
        # Normalized path
        norm_sig = normalize(sig)
        t_norm = apply_tensor(norm_sig, ORIGINAL_COEFFICIENTS)
        sym_norm = compute_symmetry(normalize(t_norm), use_adaptive=True)

        # Raw path (no normalization, fixed threshold)
        t_raw = apply_tensor(sig, ORIGINAL_COEFFICIENTS)
        sym_raw = compute_symmetry(t_raw, use_adaptive=False)

        diff = abs(sym_norm - sym_raw)
        results[sig_name] = {'normalized': sym_norm, 'raw': sym_raw, 'diff': diff}
        print(f"{sig_name:<20} {sym_norm:>12.6f} {sym_raw:>12.6f} {diff:>12.6f}")

    return results


# =============================================================================
# TEST 5: ALTERNATIVE METRICS
# =============================================================================

def test_alternative_metrics(signals):
    """Do OTHER metrics show the same pattern?"""
    print("\n" + "="*70)
    print("TEST 5: ALTERNATIVE METRICS")
    print("="*70)
    print("Question: Do different measurements show the same groupings?")

    results = {}

    def compute_autocorr_decay(sig):
        """How fast does autocorrelation decay?"""
        autocorr = np.correlate(sig, sig, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        # Find where it drops below 0.5
        below_half = np.where(autocorr < 0.5)[0]
        return below_half[0] if len(below_half) > 0 else len(autocorr)

    def compute_spectral_entropy(sig):
        """Entropy of power spectrum."""
        fft = np.abs(np.fft.rfft(sig))**2
        fft = fft / np.sum(fft)
        fft = fft[fft > 0]
        return -np.sum(fft * np.log(fft))

    def compute_hurst_exponent(sig, max_lag=100):
        """Estimate Hurst exponent (persistence)."""
        lags = range(2, min(max_lag, len(sig)//10))
        rs = []
        for lag in lags:
            chunks = [sig[i:i+lag] for i in range(0, len(sig)-lag, lag)]
            for chunk in chunks:
                r = np.max(chunk) - np.min(chunk)
                s = np.std(chunk)
                if s > 0:
                    rs.append(r/s)
        if len(rs) > 0:
            return np.mean(rs)
        return 0

    print(f"\n{'Signal':<20} {'Symmetry':>10} {'AutoCorr':>10} {'SpectEnt':>10} {'Hurst':>10}")
    print("-" * 70)

    for sig_name, sig in signals.items():
        norm_sig = normalize(sig)
        transformed = apply_tensor(norm_sig, ORIGINAL_COEFFICIENTS)
        t_norm = normalize(transformed)

        sym = compute_symmetry(t_norm)
        autocorr = compute_autocorr_decay(t_norm)
        entropy = compute_spectral_entropy(t_norm)
        hurst = compute_hurst_exponent(t_norm)

        results[sig_name] = {
            'symmetry': sym,
            'autocorr_decay': autocorr,
            'spectral_entropy': entropy,
            'hurst': hurst
        }
        print(f"{sig_name:<20} {sym:>10.4f} {autocorr:>10d} {entropy:>10.4f} {hurst:>10.4f}")

    return results


# =============================================================================
# TEST 6: NULL HYPOTHESIS - PURE RANDOMNESS
# =============================================================================

def test_null_hypothesis(n_trials=100):
    """What symmetry does PURE RANDOM DATA produce?"""
    print("\n" + "="*70)
    print("TEST 6: NULL HYPOTHESIS (Pure Random)")
    print("="*70)
    print(f"Question: What's the expected symmetry for pure noise? (n={n_trials})")

    symmetries = []
    for _ in range(n_trials):
        noise = np.random.randn(10000)
        transformed = apply_tensor(noise, ORIGINAL_COEFFICIENTS)
        sym = compute_symmetry(normalize(transformed))
        symmetries.append(sym)

    print(f"\nPure Gaussian Noise Results:")
    print(f"  Mean symmetry Δ: {np.mean(symmetries):.6f}")
    print(f"  Std:             {np.std(symmetries):.6f}")
    print(f"  Min:             {np.min(symmetries):.6f}")
    print(f"  Max:             {np.max(symmetries):.6f}")
    print(f"  95% CI:          [{np.percentile(symmetries, 2.5):.6f}, {np.percentile(symmetries, 97.5):.6f}]")

    return symmetries


# =============================================================================
# MAIN
# =============================================================================

def run_bias_check():
    """Run all bias tests."""
    print("=" * 70)
    print("BIAS CHECK ANALYSIS")
    print("Testing if results are real or methodological artifacts")
    print("=" * 70)

    signals = generate_test_signals()

    all_results = {}

    # Run all tests
    all_results['random_tensors'] = test_random_tensors(signals)
    all_results['coefficient_order'] = test_coefficient_order(signals)
    all_results['temporal_structure'] = test_temporal_structure(signals)
    all_results['normalization'] = test_normalization_effect(signals)
    all_results['alternative_metrics'] = test_alternative_metrics(signals)
    all_results['null_hypothesis'] = test_null_hypothesis()

    # Summary
    print("\n" + "=" * 70)
    print("BIAS CHECK SUMMARY")
    print("=" * 70)

    print("\n1. RANDOM TENSOR TEST:")
    print("   If original tensor gives SAME results as random tensors,")
    print("   then the specific coefficients don't matter (BAD).")

    print("\n2. COEFFICIENT ORDER TEST:")
    print("   If order doesn't matter, tensor structure is irrelevant (BAD).")

    print("\n3. TEMPORAL STRUCTURE TEST:")
    print("   If shuffling signal doesn't change results,")
    print("   we're just measuring statistical properties (MIXED).")

    print("\n4. NORMALIZATION TEST:")
    print("   If results only work with normalization,")
    print("   we're creating artificial similarity (BAD).")

    print("\n5. ALTERNATIVE METRICS TEST:")
    print("   If other metrics show different patterns,")
    print("   our metric choice may be biased (NEEDS REVIEW).")

    print("\n6. NULL HYPOTHESIS:")
    print("   Pure noise baseline - anything within this range is NOT significant.")

    # Save results
    results_dir = '/home/user/TEAPOT/results/bias_check'
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, 'bias_check_results.json')
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        json.dump(convert({
            'timestamp': datetime.now().isoformat(),
            'purpose': 'Bias check - testing if results are artifacts',
            'results': all_results
        }), f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    run_bias_check()
