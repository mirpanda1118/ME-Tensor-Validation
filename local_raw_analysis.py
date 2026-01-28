#!/usr/bin/env python3
"""
LOCAL RAW DATA ANALYSIS

Testing tensor on truly raw data we can access locally:
1. System entropy (/dev/urandom) - true hardware randomness
2. System files as raw byte streams
3. Process timing jitter
4. Memory state

No network, no synthetic generation - just raw bits.

Author: TEAPOT Raw Analysis
Date: January 2026
"""

import numpy as np
from scipy import stats
import time
import os
import json
from datetime import datetime


class TensorOperator:
    def __init__(self, coefficients=None):
        if coefficients is None:
            self.coefficients = np.array([0.1, 0.2, 0.15, 0.25, 0.3,
                                         0.2, 0.15, 0.1, 0.05, 0.1])
        else:
            self.coefficients = np.array(coefficients)

    def apply(self, sig):
        return np.convolve(sig, self.coefficients, mode='valid')

    def compute_symmetry_adaptive(self, sig):
        threshold = np.std(sig) * 0.5
        peaks, troughs = [], []
        for i in range(1, len(sig) - 1):
            if sig[i] > sig[i-1] and sig[i] > sig[i+1] and sig[i] > threshold:
                peaks.append(i)
            if sig[i] < sig[i-1] and sig[i] < sig[i+1] and sig[i] < -threshold:
                troughs.append(i)
        if len(peaks) == 0 or len(troughs) == 0:
            return 1.0, {'peaks': 0, 'troughs': 0}
        ratio = min(len(peaks), len(troughs)) / max(len(peaks), len(troughs))
        return abs(1.0 - ratio), {'peaks': len(peaks), 'troughs': len(troughs)}


def normalize(sig):
    sig = np.array(sig, dtype=float)
    std = np.std(sig)
    return (sig - np.mean(sig)) / std if std > 0 else sig - np.mean(sig)


def analyze(operator, data, name):
    if len(data) < 20:
        return None
    norm = normalize(data)
    transformed = operator.apply(norm)
    if len(transformed) < 10:
        return None
    t_norm = normalize(transformed)
    sym, counts = operator.compute_symmetry_adaptive(t_norm)
    return {
        'signal_name': name,
        'n_samples': len(data),
        'symmetry_delta': float(sym),
        'peak_count': counts['peaks'],
        'trough_count': counts['troughs'],
        'skewness': float(stats.skew(t_norm)),
        'kurtosis': float(stats.kurtosis(t_norm)),
        'zcr': float(np.sum(np.diff(np.sign(t_norm)) != 0) / len(t_norm)),
    }


# =============================================================================
# RAW LOCAL DATA SOURCES
# =============================================================================

def get_urandom_bytes(n_bytes=50000):
    """True hardware entropy from /dev/urandom."""
    print(">>> /dev/urandom - Hardware entropy")
    with open('/dev/urandom', 'rb') as f:
        raw = f.read(n_bytes)
    return np.frombuffer(raw, dtype=np.uint8).astype(float)


def get_urandom_int16(n_samples=25000):
    """Hardware entropy as int16."""
    print(">>> /dev/urandom as int16")
    with open('/dev/urandom', 'rb') as f:
        raw = f.read(n_samples * 2)
    return np.frombuffer(raw, dtype=np.int16).astype(float)


def get_urandom_float32(n_samples=12500):
    """Hardware entropy reinterpreted as float32."""
    print(">>> /dev/urandom as float32 (raw bit pattern)")
    with open('/dev/urandom', 'rb') as f:
        raw = f.read(n_samples * 4)
    # This will give us some NaN/Inf - filter those
    floats = np.frombuffer(raw, dtype=np.float32)
    valid = floats[np.isfinite(floats)]
    return valid


def get_timing_jitter(n_samples=10000):
    """Measure actual CPU timing jitter."""
    print(">>> CPU timing jitter")
    times = []
    for _ in range(n_samples):
        start = time.perf_counter_ns()
        # Do a tiny operation
        _ = 1 + 1
        end = time.perf_counter_ns()
        times.append(end - start)
    return np.array(times, dtype=float)


def get_timing_jitter_loop(n_samples=10000):
    """Timing jitter from loop iterations."""
    print(">>> Loop iteration timing")
    times = []
    for i in range(n_samples):
        start = time.perf_counter_ns()
        # Variable work
        x = 0
        for j in range(i % 10):
            x += j
        end = time.perf_counter_ns()
        times.append(end - start)
    return np.array(times, dtype=float)


def get_file_bytes(filepath, max_bytes=50000):
    """Read raw bytes from a file."""
    print(f">>> Raw file bytes: {filepath}")
    try:
        with open(filepath, 'rb') as f:
            raw = f.read(max_bytes)
        return np.frombuffer(raw, dtype=np.uint8).astype(float)
    except:
        return np.array([])


def get_proc_data():
    """Read from /proc filesystem - real system state."""
    print(">>> /proc system data")
    datasets = {}

    # /proc/stat - CPU statistics
    try:
        with open('/proc/stat', 'r') as f:
            content = f.read()
        # Extract numbers
        numbers = []
        for word in content.split():
            try:
                numbers.append(float(word))
            except ValueError:
                continue
        if numbers:
            datasets['proc_stat'] = np.array(numbers)
    except:
        pass

    # /proc/meminfo - Memory state
    try:
        with open('/proc/meminfo', 'r') as f:
            content = f.read()
        numbers = []
        for word in content.split():
            try:
                numbers.append(float(word))
            except ValueError:
                continue
        if numbers:
            datasets['proc_meminfo'] = np.array(numbers)
    except:
        pass

    # /proc/interrupts
    try:
        with open('/proc/interrupts', 'r') as f:
            content = f.read()
        numbers = []
        for word in content.split():
            try:
                numbers.append(float(word))
            except ValueError:
                continue
        if numbers:
            datasets['proc_interrupts'] = np.array(numbers)
    except:
        pass

    return datasets


def get_numpy_random_states():
    """Different numpy random generators for comparison."""
    print(">>> NumPy random generators")
    datasets = {}
    n = 10000

    # Standard normal
    datasets['numpy_normal'] = np.random.randn(n)

    # Uniform
    datasets['numpy_uniform'] = np.random.uniform(0, 1, n)

    # Different RNG
    rng = np.random.Generator(np.random.PCG64())
    datasets['numpy_pcg64'] = rng.standard_normal(n)

    rng2 = np.random.Generator(np.random.MT19937())
    datasets['numpy_mt19937'] = rng2.standard_normal(n)

    # Poisson (discrete)
    datasets['numpy_poisson'] = np.random.poisson(5, n).astype(float)

    # Exponential
    datasets['numpy_exponential'] = np.random.exponential(1, n)

    # Chi-squared
    datasets['numpy_chisquare'] = np.random.chisquare(3, n)

    return datasets


def get_mathematical_sequences():
    """Pure mathematical sequences for comparison."""
    print(">>> Mathematical sequences")
    datasets = {}
    n = 10000

    # Digits of pi (as much as we can compute)
    # We'll use the Leibniz formula approximation differences
    pi_approx = []
    total = 0
    for i in range(n):
        total += ((-1)**i) / (2*i + 1)
        pi_approx.append(total * 4)
    datasets['pi_convergence'] = np.array(pi_approx)

    # Fibonacci ratios
    fib = [1, 1]
    for i in range(n-2):
        fib.append(fib[-1] + fib[-2])
    ratios = [fib[i+1]/fib[i] for i in range(len(fib)-1)]
    datasets['fibonacci_ratios'] = np.array(ratios[:n])

    # Prime gaps
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    primes = []
    num = 2
    while len(primes) < n // 10:
        if is_prime(num):
            primes.append(num)
        num += 1
    gaps = np.diff(primes)
    datasets['prime_gaps'] = gaps.astype(float)

    # Collatz sequence lengths
    def collatz_length(n):
        count = 0
        while n != 1 and count < 1000:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3*n + 1
            count += 1
        return count

    collatz = [collatz_length(i) for i in range(1, n+1)]
    datasets['collatz_lengths'] = np.array(collatz, dtype=float)

    # Logistic map (chaos)
    r = 3.9  # Chaotic regime
    x = 0.5
    logistic = []
    for _ in range(n):
        x = r * x * (1 - x)
        logistic.append(x)
    datasets['logistic_map'] = np.array(logistic)

    return datasets


# =============================================================================
# MAIN
# =============================================================================

def run_local_raw_analysis():
    """Run analysis on all local raw data sources."""

    print("=" * 70)
    print("LOCAL RAW DATA TENSOR ANALYSIS")
    print("No network, no synthetic models - just raw bits and math")
    print("=" * 70)

    operator = TensorOperator()
    all_results = []

    # Collect all data sources
    all_data = {}

    # Hardware entropy
    all_data['ENTROPY_urandom_uint8'] = get_urandom_bytes()
    all_data['ENTROPY_urandom_int16'] = get_urandom_int16()
    urandom_float = get_urandom_float32()
    if len(urandom_float) > 100:
        all_data['ENTROPY_urandom_float32'] = urandom_float

    # Timing
    all_data['TIMING_cpu_jitter'] = get_timing_jitter()
    all_data['TIMING_loop_jitter'] = get_timing_jitter_loop()

    # System files
    all_data['FILE_urandom_raw'] = get_file_bytes('/dev/urandom')

    # Check if we have any local files to analyze
    test_files = ['/etc/passwd', '/var/log/syslog', '/var/log/dmesg']
    for tf in test_files:
        data = get_file_bytes(tf)
        if len(data) > 100:
            all_data[f'FILE_{os.path.basename(tf)}'] = data

    # Proc filesystem
    proc_data = get_proc_data()
    for name, data in proc_data.items():
        if len(data) > 20:
            all_data[f'PROC_{name}'] = data

    # NumPy generators
    numpy_data = get_numpy_random_states()
    for name, data in numpy_data.items():
        all_data[f'RANDOM_{name}'] = data

    # Mathematical sequences
    math_data = get_mathematical_sequences()
    for name, data in math_data.items():
        if len(data) > 20:
            all_data[f'MATH_{name}'] = data

    # Analyze all
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for name, data in all_data.items():
        result = analyze(operator, data, name)
        if result:
            all_results.append(result)

    # Sort and display
    sorted_results = sorted(all_results, key=lambda x: x['symmetry_delta'])

    print(f"\n{'Signal':<40} {'Δ':>10} {'Peaks':>8} {'Troughs':>8} {'Kurt':>10}")
    print("-" * 80)
    for r in sorted_results:
        print(f"{r['signal_name']:<40} {r['symmetry_delta']:>10.6f} {r['peak_count']:>8} {r['trough_count']:>8} {r['kurtosis']:>10.4f}")

    # Group analysis
    print("\n" + "=" * 70)
    print("GROUPED ANALYSIS")
    print("=" * 70)

    groups = {}
    for r in all_results:
        prefix = r['signal_name'].split('_')[0]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(r['symmetry_delta'])

    print(f"\n{'Group':<20} {'Mean Δ':>12} {'Std Δ':>12} {'N':>6}")
    print("-" * 55)
    for group, syms in sorted(groups.items()):
        print(f"{group:<20} {np.mean(syms):>12.6f} {np.std(syms):>12.6f} {len(syms):>6}")

    # Save
    results_dir = '/home/user/TEAPOT/results/local_raw_analysis'
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'local_raw_results.json'), 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
            'groups': {k: {'mean': np.mean(v), 'std': np.std(v), 'n': len(v)}
                      for k, v in groups.items()}
        }, f, indent=2)

    print(f"\nResults saved to: {results_dir}/local_raw_results.json")

    return all_results


if __name__ == "__main__":
    run_local_raw_analysis()
