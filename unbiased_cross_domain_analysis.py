#!/usr/bin/env python3
"""
Unbiased Cross-Domain Tensor Analysis

Testing domains I'm choosing (not user-specified):
1. SEISMIC - Earthquake waveforms (geological)
2. FINANCIAL - Market price movements (human/economic)
3. SOLAR - Heliophysics data (astrophysical, non-GW)
4. OCEAN - Tidal/wave measurements (fluid dynamics)
5. QUANTUM - Simulated quantum measurement sequences

These are deliberately chosen to have NO obvious relationship to:
- EEG (original domain)
- Gravitational waves
- Aerospace

Author: TEAPOT Cross-Domain Analysis
Date: January 2026
"""

import numpy as np
from scipy import stats, signal
import json
import os
from datetime import datetime
import urllib.request
import warnings
warnings.filterwarnings('ignore')


class TensorOperator:
    """Fixed 10-element tensor (binary-derived)."""

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
        peaks = self._detect_peaks(sig, threshold)
        troughs = self._detect_peaks(-sig, threshold)
        if len(peaks) == 0 or len(troughs) == 0:
            return 1.0, {'peaks': 0, 'troughs': 0}
        ratio = min(len(peaks), len(troughs)) / max(len(peaks), len(troughs))
        return abs(1.0 - ratio), {'peaks': len(peaks), 'troughs': len(troughs)}

    def _detect_peaks(self, sig, threshold):
        peaks = []
        for i in range(1, len(sig) - 1):
            if sig[i] > sig[i-1] and sig[i] > sig[i+1] and sig[i] > threshold:
                peaks.append(i)
        return peaks


def normalize(sig):
    sig = np.array(sig, dtype=float)
    std = np.std(sig)
    return (sig - np.mean(sig)) / std if std > 0 else sig - np.mean(sig)


def analyze(operator, data, name):
    """Full tensor analysis."""
    norm = normalize(data)
    transformed = operator.apply(norm)
    t_norm = normalize(transformed)

    sym, counts = operator.compute_symmetry_adaptive(t_norm)
    gradient = np.gradient(t_norm)

    return {
        'signal_name': name,
        'n_samples': len(data),
        'symmetry_delta': float(sym),
        'peak_count': counts['peaks'],
        'trough_count': counts['troughs'],
        'peak_trough_ratio': counts['peaks'] / counts['troughs'] if counts['troughs'] > 0 else float('inf'),
        'skewness': float(stats.skew(t_norm)),
        'kurtosis': float(stats.kurtosis(t_norm)),
        'zcr': float(np.sum(np.diff(np.sign(t_norm)) != 0) / len(t_norm)),
        'gradient_mean': float(np.mean(np.abs(gradient))),
        'gradient_std': float(np.std(gradient))
    }


# =============================================================================
# DOMAIN 1: SEISMIC DATA
# =============================================================================

def generate_seismic_data():
    """
    Realistic earthquake and seismic waveforms.
    Based on actual seismological parameters.
    """
    print("\n>>> SEISMIC DOMAIN")
    datasets = {}
    sr = 100  # 100 Hz typical seismometer

    # 1. P-wave arrival (compression wave)
    duration = 60
    n = sr * duration
    t = np.linspace(0, duration, n)
    # P-wave: sharp onset, high frequency
    p_arrival = 20  # seconds
    p_wave = np.zeros(n)
    p_idx = int(p_arrival * sr)
    p_duration = int(5 * sr)
    p_freq = 5  # Hz
    p_env = np.exp(-np.linspace(0, 5, p_duration))
    p_wave[p_idx:p_idx+p_duration] = p_env * np.sin(2*np.pi*p_freq*np.linspace(0, 5, p_duration))
    p_wave += 0.01 * np.random.randn(n)
    datasets['p_wave_arrival'] = {'data': p_wave, 'desc': 'P-wave seismic arrival'}

    # 2. S-wave (shear wave) - slower, larger amplitude
    s_arrival = 35
    s_wave = np.zeros(n)
    s_idx = int(s_arrival * sr)
    s_duration = int(15 * sr)
    s_freq = 2  # Hz lower frequency
    s_env = np.exp(-np.linspace(0, 3, s_duration))
    s_wave[s_idx:s_idx+s_duration] = 3 * s_env * np.sin(2*np.pi*s_freq*np.linspace(0, 15, s_duration))
    s_wave += 0.02 * np.random.randn(n)
    datasets['s_wave_arrival'] = {'data': s_wave, 'desc': 'S-wave seismic arrival'}

    # 3. Complete earthquake (P + S + surface waves)
    earthquake = p_wave.copy()
    earthquake[s_idx:s_idx+s_duration] += 3 * s_env * np.sin(2*np.pi*s_freq*np.linspace(0, 15, s_duration))
    # Add surface waves (Rayleigh)
    surf_arrival = 45
    surf_idx = int(surf_arrival * sr)
    surf_actual_len = min(int(20 * sr), n - surf_idx)
    surf_freq = 0.5
    surf_env = np.exp(-np.linspace(0, 2, surf_actual_len))
    earthquake[surf_idx:surf_idx+surf_actual_len] += 5 * surf_env * np.sin(2*np.pi*surf_freq*np.linspace(0, 20, surf_actual_len))
    datasets['full_earthquake'] = {'data': earthquake, 'desc': 'Complete earthquake waveform (P+S+surface)'}

    # 4. Volcanic tremor (continuous harmonic)
    tremor_freq = 2.5
    volcanic_tremor = np.sin(2*np.pi*tremor_freq*t) * (1 + 0.3*np.sin(2*np.pi*0.1*t))
    volcanic_tremor += 0.2 * np.random.randn(n)
    datasets['volcanic_tremor'] = {'data': volcanic_tremor, 'desc': 'Volcanic harmonic tremor'}

    # 5. Microseismic noise (ocean-generated)
    # Double-frequency microseism: 0.1-0.3 Hz
    microseism = np.sin(2*np.pi*0.15*t) + 0.5*np.sin(2*np.pi*0.2*t)
    microseism += 0.3 * np.random.randn(n)
    datasets['microseismic_noise'] = {'data': microseism, 'desc': 'Ocean-generated microseismic noise'}

    return datasets


# =============================================================================
# DOMAIN 2: FINANCIAL DATA
# =============================================================================

def generate_financial_data():
    """
    Market price and trading data.
    Human-driven, non-physical system.
    """
    print("\n>>> FINANCIAL DOMAIN")
    datasets = {}

    # 1. Stock price - Geometric Brownian Motion with jumps
    n = 10000
    dt = 1/252  # Daily
    mu = 0.1  # 10% annual return
    sigma = 0.2  # 20% volatility

    # GBM
    returns = np.random.normal(mu*dt, sigma*np.sqrt(dt), n)
    # Add jumps (earnings, news)
    jump_times = np.random.choice(n, 20, replace=False)
    returns[jump_times] += np.random.choice([-1, 1], 20) * np.random.uniform(0.02, 0.08, 20)
    price = 100 * np.exp(np.cumsum(returns))
    datasets['stock_price_gbm'] = {'data': price, 'desc': 'Stock price (GBM with jumps)'}

    # 2. High-frequency trading data (tick data)
    n_ticks = 50000
    # Tick arrivals follow Poisson process
    inter_arrival = np.random.exponential(0.001, n_ticks)
    tick_changes = np.random.choice([-0.01, 0, 0.01], n_ticks, p=[0.3, 0.4, 0.3])
    hft_price = 100 + np.cumsum(tick_changes)
    datasets['hft_tick_data'] = {'data': hft_price, 'desc': 'High-frequency tick data'}

    # 3. Market volatility (VIX-like)
    n = 5000
    # Mean-reverting volatility (Ornstein-Uhlenbeck)
    theta = 0.1  # Mean reversion speed
    mu_vol = 20  # Long-term mean
    sigma_vol = 5
    vol = np.zeros(n)
    vol[0] = 15
    for i in range(1, n):
        vol[i] = vol[i-1] + theta*(mu_vol - vol[i-1]) + sigma_vol*np.random.randn()
    vol = np.maximum(vol, 5)  # Floor at 5
    datasets['volatility_index'] = {'data': vol, 'desc': 'Market volatility index (VIX-like)'}

    # 4. Cryptocurrency - more volatile, 24/7
    n = 20000
    crypto_returns = np.random.normal(0, 0.05, n)  # 5% daily vol
    # Add regime changes
    regime = np.zeros(n)
    regime[5000:8000] = 0.1  # Bull run
    regime[12000:14000] = -0.15  # Crash
    crypto_returns += regime * np.random.uniform(0.5, 1.5, n)
    crypto_price = 1000 * np.exp(np.cumsum(crypto_returns))
    datasets['crypto_price'] = {'data': crypto_price, 'desc': 'Cryptocurrency price (high volatility)'}

    # 5. Order book imbalance
    n = 10000
    # Imbalance oscillates around zero
    imbalance = np.cumsum(np.random.randn(n) * 0.1)
    imbalance = imbalance - np.mean(imbalance)
    datasets['order_book_imbalance'] = {'data': imbalance, 'desc': 'Order book bid-ask imbalance'}

    return datasets


# =============================================================================
# DOMAIN 3: SOLAR/HELIOPHYSICS
# =============================================================================

def generate_solar_data():
    """
    Solar and space weather data.
    Astrophysical but NOT gravitational waves.
    """
    print("\n>>> SOLAR/HELIOPHYSICS DOMAIN")
    datasets = {}

    # 1. Sunspot number (11-year cycle)
    n = 4000  # ~11 years of daily data
    t = np.linspace(0, 11, n)
    # Solar cycle modeled as sinusoid with asymmetric rise/fall
    cycle_phase = 2*np.pi*t/11
    sunspot = 100 * (np.sin(cycle_phase)**2 + 0.3*np.sin(2*cycle_phase))
    sunspot = np.maximum(sunspot + 20*np.random.randn(n), 0)
    datasets['sunspot_number'] = {'data': sunspot, 'desc': 'Sunspot number (11-year cycle)'}

    # 2. Solar flare X-ray flux
    n = 10000
    # Background + impulsive events
    background = 1e-6 * np.ones(n)
    flare_flux = background.copy()
    # Add flares (exponential rise, slower decay)
    n_flares = 15
    for _ in range(n_flares):
        flare_start = np.random.randint(100, n-500)
        flare_peak = np.random.uniform(1e-5, 1e-3)
        rise_time = np.random.randint(10, 50)
        decay_time = np.random.randint(50, 200)
        # Rise phase
        flare_flux[flare_start:flare_start+rise_time] += flare_peak * np.linspace(0, 1, rise_time)**2
        # Decay phase
        flare_flux[flare_start+rise_time:flare_start+rise_time+decay_time] += flare_peak * np.exp(-np.linspace(0, 5, decay_time))
    datasets['solar_flare_xray'] = {'data': flare_flux, 'desc': 'Solar flare X-ray flux'}

    # 3. Solar wind speed
    n = 5000
    # Typically 300-800 km/s with stream interactions
    base_speed = 400
    solar_wind = base_speed + 50*np.sin(2*np.pi*np.linspace(0, 10, n))  # Corotation
    # Add high-speed streams
    stream_starts = [1000, 2500, 4000]
    for start in stream_starts:
        stream_profile = 200 * np.exp(-((np.arange(n) - start)/200)**2)
        solar_wind += stream_profile
    solar_wind += 30 * np.random.randn(n)
    datasets['solar_wind_speed'] = {'data': solar_wind, 'desc': 'Solar wind speed (km/s)'}

    # 4. Geomagnetic Dst index (storm indicator)
    n = 3000
    dst = np.zeros(n)
    # Quiet time near zero
    dst += 5 * np.random.randn(n)
    # Add geomagnetic storms (negative excursions)
    storm_times = [500, 1500, 2200]
    for st in storm_times:
        storm_intensity = np.random.uniform(-100, -300)
        storm_duration = np.random.randint(100, 300)
        recovery_time = np.random.randint(200, 500)
        # Main phase (rapid decrease)
        dst[st:st+storm_duration//3] += np.linspace(0, storm_intensity, storm_duration//3)
        # Recovery phase (slow increase)
        dst[st+storm_duration//3:st+storm_duration//3+recovery_time] += storm_intensity * np.exp(-np.linspace(0, 3, recovery_time))
    datasets['geomagnetic_dst'] = {'data': dst, 'desc': 'Geomagnetic Dst storm index'}

    # 5. Cosmic ray flux (anti-correlated with solar cycle)
    cosmic_rays = 100 - 0.3 * sunspot + 5*np.random.randn(len(sunspot))
    datasets['cosmic_ray_flux'] = {'data': cosmic_rays, 'desc': 'Galactic cosmic ray flux'}

    return datasets


# =============================================================================
# DOMAIN 4: OCEAN DATA
# =============================================================================

def generate_ocean_data():
    """
    Oceanographic measurements.
    Fluid dynamics at planetary scale.
    """
    print("\n>>> OCEAN DOMAIN")
    datasets = {}
    sr = 1  # 1 Hz for wave buoys

    # 1. Ocean surface waves (Pierson-Moskowitz spectrum)
    duration = 3600  # 1 hour
    n = sr * duration
    t = np.linspace(0, duration, n)
    freqs = np.fft.rfftfreq(n, 1/sr)
    # Pierson-Moskowitz spectrum
    U = 10  # Wind speed m/s
    alpha = 0.0081
    beta = 0.74
    g = 9.81
    omega = 2*np.pi*freqs
    omega[0] = 1e-10  # Avoid div by zero
    S_pm = alpha * g**2 / omega**5 * np.exp(-beta * (g / (omega * U))**4)
    S_pm[0] = 0
    phases = np.random.uniform(0, 2*np.pi, len(freqs))
    wave_fft = np.sqrt(S_pm) * np.exp(1j * phases)
    ocean_waves = np.fft.irfft(wave_fft, n)
    datasets['ocean_surface_waves'] = {'data': ocean_waves, 'desc': 'Ocean surface waves (Pierson-Moskowitz)'}

    # 2. Tidal signal (mixed semidiurnal)
    duration = 86400 * 30  # 30 days
    n = 8640  # 10-minute resolution
    t = np.linspace(0, duration, n)
    # M2 (principal lunar): 12.42 hour period
    # S2 (principal solar): 12.00 hour period
    # K1 (lunar diurnal): 23.93 hour period
    M2 = 1.0 * np.cos(2*np.pi*t/(12.42*3600))
    S2 = 0.5 * np.cos(2*np.pi*t/(12.00*3600))
    K1 = 0.3 * np.cos(2*np.pi*t/(23.93*3600))
    tide = M2 + S2 + K1 + 0.05*np.random.randn(n)
    datasets['tidal_signal'] = {'data': tide, 'desc': 'Mixed semidiurnal tidal signal'}

    # 3. Tsunami waveform
    n = 10000
    t = np.linspace(0, 10000, n)  # seconds
    # Long period wave, dispersive
    tsunami = np.zeros(n)
    arrival = 2000
    # First wave
    tsunami[arrival:arrival+1000] = 2 * np.sin(np.pi*np.linspace(0, 1, 1000))
    # Subsequent waves (dispersive, period decreases)
    for i, offset in enumerate([3500, 4800, 5900]):
        period_factor = 1 - 0.2*i
        wave_len = int(1000 * period_factor)
        if offset + wave_len < n:
            tsunami[offset:offset+wave_len] = (1.5 - 0.3*i) * np.sin(np.pi*np.linspace(0, 1, wave_len))
    tsunami += 0.1 * np.random.randn(n)
    datasets['tsunami_waveform'] = {'data': tsunami, 'desc': 'Tsunami wave train (dispersive)'}

    # 4. Ocean internal waves
    n = 5000
    t = np.linspace(0, 5000, n)
    # Internal waves: periods of minutes to hours
    internal = np.sin(2*np.pi*t/600) + 0.5*np.sin(2*np.pi*t/1800)
    internal += 0.2 * np.random.randn(n)
    datasets['internal_waves'] = {'data': internal, 'desc': 'Ocean internal waves'}

    # 5. Rogue wave event
    n = 5000
    t = np.linspace(0, 5000, n)
    # Background sea state
    rogue = np.random.randn(n) * 2  # 2m significant wave height
    # Add rogue wave (H > 2*Hs)
    rogue_time = 2500
    rogue_height = 10  # 10m wave
    rogue_width = 50
    rogue[rogue_time:rogue_time+rogue_width] += rogue_height * np.sin(np.pi*np.linspace(0, 1, rogue_width))
    datasets['rogue_wave'] = {'data': rogue, 'desc': 'Rogue wave event in sea state'}

    return datasets


# =============================================================================
# DOMAIN 5: QUANTUM-LIKE MEASUREMENTS
# =============================================================================

def generate_quantum_data():
    """
    Simulated quantum measurement sequences.
    Fundamentally probabilistic.
    """
    print("\n>>> QUANTUM DOMAIN")
    datasets = {}

    # 1. Qubit measurement sequence (|0> vs |1>)
    n = 10000
    # Superposition state |psi> = cos(theta)|0> + sin(theta)|1>
    theta = np.pi/4  # Equal superposition
    prob_1 = np.sin(theta)**2
    measurements = np.random.choice([0, 1], n, p=[1-prob_1, prob_1])
    datasets['qubit_measurements'] = {'data': measurements.astype(float), 'desc': 'Qubit measurement sequence (superposition)'}

    # 2. Quantum random walk
    n = 5000
    # Hadamard walk on a line
    position = 0
    positions = [position]
    for _ in range(n-1):
        # Coin flip (Hadamard-like)
        if np.random.random() < 0.5:
            position += 1
        else:
            position -= 1
        # Quantum interference (biased toward certain positions)
        if position % 2 == 0:
            position += np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
        positions.append(position)
    datasets['quantum_walk'] = {'data': np.array(positions, dtype=float), 'desc': 'Quantum random walk position'}

    # 3. Photon counting (coherent state)
    n = 5000
    # Poisson distribution for coherent light
    mean_photons = 10
    photon_counts = np.random.poisson(mean_photons, n)
    datasets['coherent_photon_counting'] = {'data': photon_counts.astype(float), 'desc': 'Coherent state photon counting'}

    # 4. Squeezed state measurements
    n = 5000
    # Squeezed vacuum: reduced noise in one quadrature
    squeeze_factor = 0.3
    x_quadrature = np.random.normal(0, squeeze_factor, n)
    p_quadrature = np.random.normal(0, 1/squeeze_factor, n)
    # Measure x quadrature
    datasets['squeezed_state'] = {'data': x_quadrature, 'desc': 'Squeezed vacuum X-quadrature'}

    # 5. Bell test correlations
    n = 5000
    # Simulated Bell state measurements
    # |Phi+> = (|00> + |11>)/sqrt(2)
    # Perfect correlations when measured in same basis
    alice_basis = np.random.choice([0, 1], n)  # 0 = Z basis, 1 = X basis
    bob_basis = np.random.choice([0, 1], n)

    correlations = np.zeros(n)
    for i in range(n):
        if alice_basis[i] == bob_basis[i]:
            # Same basis: perfect correlation
            correlations[i] = 1
        else:
            # Different basis: random
            correlations[i] = np.random.choice([-1, 1])
    datasets['bell_correlations'] = {'data': correlations, 'desc': 'Bell state measurement correlations'}

    return datasets


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_unbiased_analysis():
    """Run tensor analysis on all unbiased domains."""

    print("=" * 80)
    print("UNBIASED CROSS-DOMAIN TENSOR ANALYSIS")
    print("Domains selected by AI, not user-specified")
    print("=" * 80)

    operator = TensorOperator()
    all_results = []

    # Collect all domains
    domains = {
        'SEISMIC': generate_seismic_data(),
        'FINANCIAL': generate_financial_data(),
        'SOLAR': generate_solar_data(),
        'OCEAN': generate_ocean_data(),
        'QUANTUM': generate_quantum_data()
    }

    # Analyze each
    print("\n" + "=" * 80)
    print("TENSOR ANALYSIS RESULTS")
    print("=" * 80)

    for domain_name, datasets in domains.items():
        print(f"\n{'='*40}")
        print(f"  {domain_name}")
        print(f"{'='*40}")

        for sig_name, sig_data in datasets.items():
            full_name = f"{domain_name}_{sig_name}"
            result = analyze(operator, sig_data['data'], full_name)
            result['description'] = sig_data['desc']
            result['domain'] = domain_name
            all_results.append(result)

            print(f"\n  {sig_name}:")
            print(f"    {sig_data['desc']}")
            print(f"    Symmetry Δ: {result['symmetry_delta']:.6f}")
            print(f"    Peak/Trough: {result['peak_count']}/{result['trough_count']} ({result['peak_trough_ratio']:.3f})")
            print(f"    Kurtosis: {result['kurtosis']:.4f}, ZCR: {result['zcr']:.6f}")

    # Cross-domain comparison
    print("\n" + "=" * 80)
    print("CROSS-DOMAIN COMPARISON")
    print("=" * 80)

    # By domain
    print("\n1. SYMMETRY BY DOMAIN:")
    for domain in ['SEISMIC', 'FINANCIAL', 'SOLAR', 'OCEAN', 'QUANTUM']:
        domain_results = [r for r in all_results if r['domain'] == domain]
        syms = [r['symmetry_delta'] for r in domain_results]
        print(f"   {domain:12s}: mean={np.mean(syms):.6f}, std={np.std(syms):.6f}, range=[{min(syms):.4f}, {max(syms):.4f}]")

    print("\n2. OVERALL RANKING (lowest Δ = most symmetric):")
    sorted_results = sorted(all_results, key=lambda x: x['symmetry_delta'])
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"   {i:2d}. {r['signal_name']:40s}: Δ = {r['symmetry_delta']:.6f}")

    print("\n   ...")
    print("\n   MOST ASYMMETRIC:")
    for r in sorted_results[-5:]:
        print(f"       {r['signal_name']:40s}: Δ = {r['symmetry_delta']:.6f}")

    # Statistical comparison to GW results
    print("\n3. COMPARISON TO GRAVITATIONAL WAVE BASELINE:")
    print("   (GW events showed: Δ = 0.027 ± 0.015, ZCR = 0.028 ± 0.003)")

    gw_like = [r for r in all_results if r['symmetry_delta'] < 0.05 and r['zcr'] < 0.05]
    print(f"\n   Signals with GW-like response (Δ < 0.05, ZCR < 0.05):")
    for r in gw_like:
        print(f"     {r['signal_name']:40s}: Δ={r['symmetry_delta']:.4f}, ZCR={r['zcr']:.4f}")

    # Save results
    results_dir = '/home/user/TEAPOT/results/unbiased_analysis'
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, 'unbiased_cross_domain_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'methodology': 'AI-selected unbiased domains',
            'domains_tested': list(domains.keys()),
            'results': all_results
        }, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    run_unbiased_analysis()
