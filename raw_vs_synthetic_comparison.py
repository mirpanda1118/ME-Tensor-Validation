#!/usr/bin/env python3
"""
DIRECT COMPARISON: Raw vs Synthetic

Side-by-side test of tensor on:
- RAW: Actual unprocessed data from real sources
- SYNTHETIC: Mathematically generated approximations

The question: Does the tensor respond differently?
"""

import numpy as np
from scipy import stats
import time
import os

# Tensor
class Tensor:
    def __init__(self):
        self.c = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.2, 0.15, 0.1, 0.05, 0.1])

    def analyze(self, data):
        data = np.array(data, dtype=float)
        # Normalize
        data = (data - np.mean(data)) / (np.std(data) + 1e-10)
        # Apply
        t = np.convolve(data, self.c, mode='valid')
        t = (t - np.mean(t)) / (np.std(t) + 1e-10)
        # Symmetry
        thresh = np.std(t) * 0.5
        peaks = sum(1 for i in range(1, len(t)-1)
                   if t[i] > t[i-1] and t[i] > t[i+1] and t[i] > thresh)
        troughs = sum(1 for i in range(1, len(t)-1)
                     if t[i] < t[i-1] and t[i] < t[i+1] and t[i] < -thresh)
        if peaks == 0 or troughs == 0:
            sym = 1.0
        else:
            sym = abs(1.0 - min(peaks,troughs)/max(peaks,troughs))
        return {
            'symmetry': sym,
            'peaks': peaks,
            'troughs': troughs,
            'kurtosis': float(stats.kurtosis(t)),
            'zcr': float(np.sum(np.diff(np.sign(t)) != 0) / len(t))
        }

tensor = Tensor()

print("=" * 70)
print("RAW vs SYNTHETIC - Direct Tensor Comparison")
print("=" * 70)

results = []

# =============================================================================
# TEST 1: RANDOMNESS
# =============================================================================
print("\n" + "=" * 70)
print("TEST 1: RANDOMNESS")
print("=" * 70)

# RAW: Hardware entropy
with open('/dev/urandom', 'rb') as f:
    raw_entropy = np.frombuffer(f.read(40000), dtype=np.uint8).astype(float)
r1 = tensor.analyze(raw_entropy)
print(f"\nRAW /dev/urandom (hardware):  Δ = {r1['symmetry']:.6f}  peaks={r1['peaks']} troughs={r1['troughs']}")

# SYNTHETIC: numpy random
synth_random = np.random.randint(0, 256, 40000).astype(float)
r2 = tensor.analyze(synth_random)
print(f"SYNTHETIC np.random.randint:  Δ = {r2['symmetry']:.6f}  peaks={r2['peaks']} troughs={r2['troughs']}")

# SYNTHETIC: Gaussian
synth_gauss = np.random.randn(40000) * 128 + 128
r3 = tensor.analyze(synth_gauss)
print(f"SYNTHETIC np.random.randn:    Δ = {r3['symmetry']:.6f}  peaks={r3['peaks']} troughs={r3['troughs']}")

results.append(('RANDOM_raw_entropy', r1['symmetry'], 'RAW'))
results.append(('RANDOM_synth_uniform', r2['symmetry'], 'SYNTHETIC'))
results.append(('RANDOM_synth_gaussian', r3['symmetry'], 'SYNTHETIC'))

# =============================================================================
# TEST 2: TIMING / JITTER
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: TIMING JITTER")
print("=" * 70)

# RAW: Actual CPU timing
raw_timing = []
for _ in range(10000):
    t1 = time.perf_counter_ns()
    x = sum(range(50))  # Some work
    t2 = time.perf_counter_ns()
    raw_timing.append(t2 - t1)
raw_timing = np.array(raw_timing, dtype=float)
r4 = tensor.analyze(raw_timing)
print(f"\nRAW CPU timing jitter:        Δ = {r4['symmetry']:.6f}  peaks={r4['peaks']} troughs={r4['troughs']}")

# SYNTHETIC: Modeled timing (exponential + noise)
synth_timing = np.random.exponential(np.mean(raw_timing), 10000)
synth_timing += np.random.randn(10000) * np.std(raw_timing) * 0.1
r5 = tensor.analyze(synth_timing)
print(f"SYNTHETIC exponential model:  Δ = {r5['symmetry']:.6f}  peaks={r5['peaks']} troughs={r5['troughs']}")

results.append(('TIMING_raw_cpu', r4['symmetry'], 'RAW'))
results.append(('TIMING_synth_exp', r5['symmetry'], 'SYNTHETIC'))

# =============================================================================
# TEST 3: OSCILLATIONS
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: OSCILLATIONS")
print("=" * 70)

# RAW: Can't get real oscillation data, but we can create from raw entropy
# Convert entropy to "oscillation" by cumsum and detrend
raw_walk = np.cumsum(raw_entropy - 128)  # Random walk from raw bits
raw_walk = raw_walk - np.linspace(raw_walk[0], raw_walk[-1], len(raw_walk))  # Detrend
r6 = tensor.analyze(raw_walk)
print(f"\nRAW entropy random walk:      Δ = {r6['symmetry']:.6f}  peaks={r6['peaks']} troughs={r6['troughs']}")

# SYNTHETIC: Clean sine wave
t = np.linspace(0, 100, 40000)
synth_sine = np.sin(2 * np.pi * t)
r7 = tensor.analyze(synth_sine)
print(f"SYNTHETIC pure sine:          Δ = {r7['symmetry']:.6f}  peaks={r7['peaks']} troughs={r7['troughs']}")

# SYNTHETIC: Noisy sine
synth_sine_noisy = synth_sine + 0.2 * np.random.randn(40000)
r8 = tensor.analyze(synth_sine_noisy)
print(f"SYNTHETIC noisy sine:         Δ = {r8['symmetry']:.6f}  peaks={r8['peaks']} troughs={r8['troughs']}")

# SYNTHETIC: Chirp (GW-like)
f = 1 + 10 * t / 100
synth_chirp = np.sin(2 * np.pi * np.cumsum(f) / 400)
r9 = tensor.analyze(synth_chirp)
print(f"SYNTHETIC chirp (GW-like):    Δ = {r9['symmetry']:.6f}  peaks={r9['peaks']} troughs={r9['troughs']}")

results.append(('OSCILLATION_raw_walk', r6['symmetry'], 'RAW'))
results.append(('OSCILLATION_synth_sine', r7['symmetry'], 'SYNTHETIC'))
results.append(('OSCILLATION_synth_noisy', r8['symmetry'], 'SYNTHETIC'))
results.append(('OSCILLATION_synth_chirp', r9['symmetry'], 'SYNTHETIC'))

# =============================================================================
# TEST 4: STRUCTURED DATA
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: STRUCTURED DATA")
print("=" * 70)

# RAW: /proc/interrupts (real kernel counters)
try:
    with open('/proc/interrupts', 'r') as f:
        content = f.read()
    raw_ints = []
    for word in content.split():
        try:
            raw_ints.append(float(word))
        except:
            pass
    if len(raw_ints) > 100:
        raw_ints = np.array(raw_ints[:10000])
        r10 = tensor.analyze(raw_ints)
        print(f"\nRAW /proc/interrupts:         Δ = {r10['symmetry']:.6f}  peaks={r10['peaks']} troughs={r10['troughs']}")
        results.append(('STRUCT_raw_interrupts', r10['symmetry'], 'RAW'))
except Exception as e:
    print(f"\nRAW /proc/interrupts: {e}")

# RAW: File bytes (structured binary)
try:
    with open('/bin/ls', 'rb') as f:
        raw_binary = np.frombuffer(f.read(40000), dtype=np.uint8).astype(float)
    r11 = tensor.analyze(raw_binary)
    print(f"RAW /bin/ls binary:           Δ = {r11['symmetry']:.6f}  peaks={r11['peaks']} troughs={r11['troughs']}")
    results.append(('STRUCT_raw_binary', r11['symmetry'], 'RAW'))
except Exception as e:
    print(f"RAW binary: {e}")

# SYNTHETIC: Modeled structured (step functions + noise)
synth_struct = np.zeros(10000)
for i in range(0, 10000, 500):
    synth_struct[i:i+500] = np.random.randint(0, 256)
synth_struct += np.random.randn(10000) * 10
r12 = tensor.analyze(synth_struct)
print(f"SYNTHETIC step+noise:         Δ = {r12['symmetry']:.6f}  peaks={r12['peaks']} troughs={r12['troughs']}")
results.append(('STRUCT_synth_steps', r12['symmetry'], 'SYNTHETIC'))

# =============================================================================
# TEST 5: CHAOTIC SYSTEMS
# =============================================================================
print("\n" + "=" * 70)
print("TEST 5: CHAOS")
print("=" * 70)

# RAW: Chaos from entropy-seeded logistic map
seed = raw_entropy[0] / 256
x = seed
raw_chaos = []
for _ in range(10000):
    x = 3.9 * x * (1 - x)
    raw_chaos.append(x)
raw_chaos = np.array(raw_chaos)
r13 = tensor.analyze(raw_chaos)
print(f"\nRAW-seeded logistic map:      Δ = {r13['symmetry']:.6f}  peaks={r13['peaks']} troughs={r13['troughs']}")

# SYNTHETIC: Fixed-seed logistic map
x = 0.5
synth_chaos = []
for _ in range(10000):
    x = 3.9 * x * (1 - x)
    synth_chaos.append(x)
synth_chaos = np.array(synth_chaos)
r14 = tensor.analyze(synth_chaos)
print(f"SYNTHETIC logistic (x0=0.5):  Δ = {r14['symmetry']:.6f}  peaks={r14['peaks']} troughs={r14['troughs']}")

results.append(('CHAOS_raw_seeded', r13['symmetry'], 'RAW'))
results.append(('CHAOS_synth_fixed', r14['symmetry'], 'SYNTHETIC'))

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY: RAW vs SYNTHETIC")
print("=" * 70)

raw_results = [(n, s) for n, s, t in results if t == 'RAW']
synth_results = [(n, s) for n, s, t in results if t == 'SYNTHETIC']

print(f"\n{'Signal':<35} {'Type':<12} {'Symmetry Δ':>12}")
print("-" * 60)
for name, sym, typ in sorted(results, key=lambda x: x[1]):
    print(f"{name:<35} {typ:<12} {sym:>12.6f}")

print("\n" + "-" * 60)
raw_syms = [s for _, s in raw_results]
synth_syms = [s for _, s in synth_results]

print(f"\nRAW data:       mean Δ = {np.mean(raw_syms):.6f} ± {np.std(raw_syms):.6f}")
print(f"SYNTHETIC data: mean Δ = {np.mean(synth_syms):.6f} ± {np.std(synth_syms):.6f}")

# Statistical test
if len(raw_syms) >= 3 and len(synth_syms) >= 3:
    t_stat, p_val = stats.ttest_ind(raw_syms, synth_syms)
    print(f"\nt-test: t = {t_stat:.4f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("SIGNIFICANT DIFFERENCE between raw and synthetic")
    else:
        print("NO SIGNIFICANT DIFFERENCE between raw and synthetic")

print("\n" + "=" * 70)
