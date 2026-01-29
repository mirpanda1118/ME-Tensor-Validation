#!/usr/bin/env python3
"""
TENSOR BEHAVIOR STUDY - Raw Data Only

One pass. Real data. No interpretation.
"""

import numpy as np
from scipy import stats

# The tensor (binary-derived, fixed)
C = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.2, 0.15, 0.1, 0.05, 0.1])

def tensor_response(data):
    """Apply tensor, return all metrics."""
    d = np.asarray(data, dtype=float)
    d = (d - d.mean()) / (d.std() + 1e-10)
    t = np.convolve(d, C, mode='valid')
    t = (t - t.mean()) / (t.std() + 1e-10)

    th = t.std() * 0.5
    p = sum(1 for i in range(1,len(t)-1) if t[i]>t[i-1] and t[i]>t[i+1] and t[i]>th)
    tr = sum(1 for i in range(1,len(t)-1) if t[i]<t[i-1] and t[i]<t[i+1] and t[i]<-th)

    sym = 1.0 if p==0 or tr==0 else abs(1 - min(p,tr)/max(p,tr))

    return {
        'n': len(data),
        'sym': round(sym, 6),
        'p/t': f"{p}/{tr}",
        'kurt': round(stats.kurtosis(t), 2),
        'zcr': round(np.sum(np.diff(np.sign(t))!=0)/len(t), 4)
    }

# Collect all real data
DATA = {}

print("Collecting real data...\n")

# 1. GW - pycbc numerical relativity
try:
    from pycbc import waveform
    hp, _ = waveform.get_td_waveform(approximant='SEOBNRv4', mass1=36, mass2=29, delta_t=1/4096, f_lower=20)
    DATA['GW150914 (pycbc)'] = np.array(hp)
    print(f"✓ GW150914: {len(hp)} samples")
except Exception as e:
    print(f"✗ GW: {e}")

# 2. Seismic - obspy
try:
    from obspy import read
    st = read()  # Built-in example
    DATA['Seismic (obspy)'] = st[0].data
    print(f"✓ Seismic: {len(st[0].data)} samples")
except Exception as e:
    print(f"✗ Seismic: {e}")

# 3. Hardware entropy
try:
    with open('/dev/urandom', 'rb') as f:
        DATA['Entropy (hardware)'] = np.frombuffer(f.read(20000), dtype=np.int16).astype(float)
    print(f"✓ Entropy: {len(DATA['Entropy (hardware)'])} samples")
except Exception as e:
    print(f"✗ Entropy: {e}")

# 4. CPU timing (raw system behavior)
import time
t_data = []
for _ in range(5000):
    t1 = time.perf_counter_ns()
    _ = sum(range(10))
    t_data.append(time.perf_counter_ns() - t1)
DATA['CPU timing (raw)'] = np.array(t_data, dtype=float)
print(f"✓ CPU timing: {len(t_data)} samples")

# 5. Binary file structure
try:
    with open('/bin/ls', 'rb') as f:
        DATA['Binary /bin/ls'] = np.frombuffer(f.read(20000), dtype=np.uint8).astype(float)
    print(f"✓ Binary: {len(DATA['Binary /bin/ls'])} samples")
except:
    pass

# 6. Mathematical sequences (deterministic, for comparison)
x = 0.5
logistic = [x := 3.9*x*(1-x) for _ in range(5000)]
DATA['Logistic map (chaos)'] = np.array(logistic)
print(f"✓ Logistic: 5000 samples")

# Run tensor on all
print("\n" + "="*75)
print("TENSOR BEHAVIOR ON RAW DATA")
print("="*75)
print(f"\n{'Source':<25} {'N':>8} {'Sym Δ':>10} {'P/T':>12} {'Kurt':>8} {'ZCR':>8}")
print("-"*75)

results = []
for name, data in DATA.items():
    r = tensor_response(data)
    results.append((name, r))
    print(f"{name:<25} {r['n']:>8} {r['sym']:>10} {r['p/t']:>12} {r['kurt']:>8} {r['zcr']:>8}")

# Sort by symmetry
print("\n" + "-"*75)
print("RANKED BY SYMMETRY (low Δ = more symmetric)")
print("-"*75)
for name, r in sorted(results, key=lambda x: x[1]['sym']):
    print(f"  {r['sym']:.6f}  {name}")

print("\n" + "="*75)
