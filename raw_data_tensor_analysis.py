#!/usr/bin/env python3
"""
RAW DATA TENSOR ANALYSIS

Fetching ACTUAL unfiltered data from real sources - no synthetic generation.

Sources:
1. USGS Earthquake - real seismometer data
2. NOAA - real ocean buoy data
3. NOAA Space Weather - real solar data
4. Yahoo Finance - real market data

Author: TEAPOT Raw Data Analysis
Date: January 2026
"""

import numpy as np
from scipy import stats
import json
import os
from datetime import datetime
import urllib.request
import csv
import io
import warnings
warnings.filterwarnings('ignore')


class TensorOperator:
    """Fixed 10-element tensor."""
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
    gradient = np.gradient(t_norm)
    return {
        'signal_name': name,
        'n_samples': len(data),
        'symmetry_delta': float(sym),
        'peak_count': counts['peaks'],
        'trough_count': counts['troughs'],
        'skewness': float(stats.skew(t_norm)),
        'kurtosis': float(stats.kurtosis(t_norm)),
        'zcr': float(np.sum(np.diff(np.sign(t_norm)) != 0) / len(t_norm)),
        'gradient_mean': float(np.mean(np.abs(gradient))),
        'data_source': 'RAW_UNFILTERED'
    }


# =============================================================================
# REAL DATA FETCHERS
# =============================================================================

def fetch_usgs_earthquake_data():
    """
    Fetch REAL earthquake data from USGS.
    Returns raw seismic measurements.
    """
    print("\n>>> Fetching REAL USGS Earthquake Data...")
    datasets = {}

    # USGS provides earthquake catalog - we'll get magnitude/depth time series
    # This is real recorded data
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2024-01-01&endtime=2024-01-31&minmagnitude=2.5&orderby=time"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(data))
            rows = list(reader)

            if len(rows) > 100:
                # Extract magnitude time series (raw values as recorded)
                magnitudes = [float(r['mag']) for r in rows if r['mag']]
                depths = [float(r['depth']) for r in rows if r['depth']]
                latitudes = [float(r['latitude']) for r in rows if r['latitude']]

                datasets['usgs_magnitudes'] = {
                    'data': np.array(magnitudes),
                    'desc': f'USGS earthquake magnitudes (n={len(magnitudes)}, raw catalog)'
                }
                datasets['usgs_depths'] = {
                    'data': np.array(depths),
                    'desc': f'USGS earthquake depths km (n={len(depths)}, raw catalog)'
                }
                datasets['usgs_latitudes'] = {
                    'data': np.array(latitudes),
                    'desc': f'USGS earthquake latitudes (n={len(latitudes)}, raw catalog)'
                }
                print(f"    Fetched {len(rows)} earthquake records")
            else:
                print(f"    Only {len(rows)} records found")

    except Exception as e:
        print(f"    Error: {e}")

    return datasets


def fetch_noaa_buoy_data():
    """
    Fetch REAL ocean buoy data from NDBC (National Data Buoy Center).
    Raw wave height, period, temperature measurements.
    """
    print("\n>>> Fetching REAL NOAA Buoy Data...")
    datasets = {}

    # Station 46222 - San Pedro, CA (commonly available)
    # Real-time data in standard meteorological format
    buoy_id = "46222"
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            lines = response.read().decode('utf-8').strip().split('\n')

            # Skip header lines (start with #)
            data_lines = [l for l in lines if not l.startswith('#')]

            if len(data_lines) > 10:
                # Parse columns: WVHT (wave height), DPD (dominant period), WTMP (water temp)
                wave_heights = []
                wave_periods = []
                water_temps = []

                for line in data_lines[1:]:  # Skip column header
                    parts = line.split()
                    if len(parts) >= 12:
                        try:
                            wvht = float(parts[8]) if parts[8] != 'MM' else None
                            dpd = float(parts[9]) if parts[9] != 'MM' else None
                            wtmp = float(parts[14]) if len(parts) > 14 and parts[14] != 'MM' else None

                            if wvht and wvht < 99:
                                wave_heights.append(wvht)
                            if dpd and dpd < 99:
                                wave_periods.append(dpd)
                            if wtmp and wtmp < 99:
                                water_temps.append(wtmp)
                        except (ValueError, IndexError):
                            continue

                if wave_heights:
                    datasets['noaa_wave_height'] = {
                        'data': np.array(wave_heights),
                        'desc': f'NOAA buoy wave height meters (n={len(wave_heights)}, raw sensor)'
                    }
                if wave_periods:
                    datasets['noaa_wave_period'] = {
                        'data': np.array(wave_periods),
                        'desc': f'NOAA buoy wave period seconds (n={len(wave_periods)}, raw sensor)'
                    }
                if water_temps:
                    datasets['noaa_water_temp'] = {
                        'data': np.array(water_temps),
                        'desc': f'NOAA buoy water temp C (n={len(water_temps)}, raw sensor)'
                    }

                print(f"    Fetched buoy data: {len(wave_heights)} wave heights, {len(wave_periods)} periods")

    except Exception as e:
        print(f"    Error fetching buoy {buoy_id}: {e}")

    return datasets


def fetch_noaa_space_weather():
    """
    Fetch REAL space weather data from NOAA SWPC.
    Solar X-ray flux, proton flux, geomagnetic indices.
    """
    print("\n>>> Fetching REAL NOAA Space Weather Data...")
    datasets = {}

    # GOES X-ray flux (1-minute data)
    url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))

            if data:
                # Extract X-ray flux values (raw measurements)
                xray_short = [d['flux'] for d in data if 'flux' in d and d.get('energy') == '0.05-0.4nm']
                xray_long = [d['flux'] for d in data if 'flux' in d and d.get('energy') == '0.1-0.8nm']

                if xray_short:
                    datasets['goes_xray_short'] = {
                        'data': np.array(xray_short),
                        'desc': f'GOES X-ray 0.05-0.4nm (n={len(xray_short)}, raw satellite)'
                    }
                if xray_long:
                    datasets['goes_xray_long'] = {
                        'data': np.array(xray_long),
                        'desc': f'GOES X-ray 0.1-0.8nm (n={len(xray_long)}, raw satellite)'
                    }

                print(f"    Fetched {len(xray_short)} X-ray measurements")

    except Exception as e:
        print(f"    Error fetching X-ray: {e}")

    # Geomagnetic Kp index
    url2 = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    try:
        with urllib.request.urlopen(url2, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data:
                kp_values = [d['kp_index'] for d in data if 'kp_index' in d]
                if kp_values:
                    datasets['geomagnetic_kp'] = {
                        'data': np.array(kp_values),
                        'desc': f'Planetary Kp index (n={len(kp_values)}, raw measurement)'
                    }
                    print(f"    Fetched {len(kp_values)} Kp index values")
    except Exception as e:
        print(f"    Error fetching Kp: {e}")

    return datasets


def fetch_financial_data():
    """
    Fetch REAL financial data.
    Using public APIs for actual market data.
    """
    print("\n>>> Fetching REAL Financial Data...")
    datasets = {}

    # Try Alpha Vantage demo or Yahoo Finance via yfinance-style URL
    # Using a simple CSV endpoint that's publicly available

    # FRED (Federal Reserve Economic Data) - public API
    # 10-Year Treasury Rate
    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd=2023-01-01&coed=2024-01-01"

    try:
        with urllib.request.urlopen(fred_url, timeout=30) as response:
            data = response.read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(data))
            rows = list(reader)

            values = []
            for r in rows:
                try:
                    val = float(r['DGS10'])
                    values.append(val)
                except (ValueError, KeyError):
                    continue

            if values:
                datasets['treasury_10yr'] = {
                    'data': np.array(values),
                    'desc': f'10-Year Treasury Rate (n={len(values)}, FRED raw)'
                }
                print(f"    Fetched {len(values)} Treasury rate values")

    except Exception as e:
        print(f"    Error fetching FRED: {e}")

    # Try fetching Bitcoin price from a public API
    btc_url = "https://api.coindesk.com/v1/bpi/historical/close.json?start=2023-01-01&end=2024-01-01"

    try:
        with urllib.request.urlopen(btc_url, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            if 'bpi' in data:
                prices = list(data['bpi'].values())
                datasets['bitcoin_price'] = {
                    'data': np.array(prices),
                    'desc': f'Bitcoin daily price USD (n={len(prices)}, raw market)'
                }
                print(f"    Fetched {len(prices)} Bitcoin prices")
    except Exception as e:
        print(f"    Error fetching Bitcoin: {e}")

    return datasets


def fetch_climate_data():
    """
    Fetch REAL climate/atmospheric data.
    """
    print("\n>>> Fetching REAL Climate Data...")
    datasets = {}

    # NOAA Climate Data - Global temperature anomalies
    url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/1/0/1880-2024/data.csv"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode('utf-8')
            lines = data.strip().split('\n')

            # Skip header lines
            values = []
            for line in lines:
                if line and not line.startswith('Global') and ',' in line:
                    parts = line.split(',')
                    try:
                        val = float(parts[1])
                        values.append(val)
                    except (ValueError, IndexError):
                        continue

            if values:
                datasets['global_temp_anomaly'] = {
                    'data': np.array(values),
                    'desc': f'Global temperature anomaly C (n={len(values)}, NOAA raw)'
                }
                print(f"    Fetched {len(values)} temperature anomaly values")

    except Exception as e:
        print(f"    Error fetching climate: {e}")

    return datasets


def generate_local_raw_noise():
    """
    Generate raw noise from system entropy for comparison.
    This is as 'raw' as we can get computationally.
    """
    print("\n>>> Generating System Entropy Noise (baseline)...")
    datasets = {}

    # Use /dev/urandom for true system randomness
    try:
        with open('/dev/urandom', 'rb') as f:
            raw_bytes = f.read(10000)
            # Convert to float array
            raw_ints = np.frombuffer(raw_bytes, dtype=np.uint8)
            datasets['system_entropy'] = {
                'data': raw_ints.astype(float),
                'desc': 'System entropy (/dev/urandom, true random)'
            }
            print(f"    Generated {len(raw_ints)} entropy samples")
    except:
        # Fallback
        datasets['system_entropy'] = {
            'data': np.random.randint(0, 256, 10000).astype(float),
            'desc': 'Pseudo-random entropy (fallback)'
        }

    return datasets


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_raw_data_analysis():
    """Analyze REAL raw data vs synthetic baseline."""

    print("=" * 70)
    print("RAW UNFILTERED DATA TENSOR ANALYSIS")
    print("Testing tensor on ACTUAL data from real instruments")
    print("=" * 70)

    operator = TensorOperator()
    all_results = []

    # Fetch all real data
    all_datasets = {}

    earthquake_data = fetch_usgs_earthquake_data()
    buoy_data = fetch_noaa_buoy_data()
    space_data = fetch_noaa_space_weather()
    financial_data = fetch_financial_data()
    climate_data = fetch_climate_data()
    entropy_data = generate_local_raw_noise()

    all_datasets.update({f"EARTHQUAKE_{k}": v for k, v in earthquake_data.items()})
    all_datasets.update({f"OCEAN_{k}": v for k, v in buoy_data.items()})
    all_datasets.update({f"SOLAR_{k}": v for k, v in space_data.items()})
    all_datasets.update({f"FINANCIAL_{k}": v for k, v in financial_data.items()})
    all_datasets.update({f"CLIMATE_{k}": v for k, v in climate_data.items()})
    all_datasets.update({f"BASELINE_{k}": v for k, v in entropy_data.items()})

    # Analyze each
    print("\n" + "=" * 70)
    print("TENSOR ANALYSIS ON RAW DATA")
    print("=" * 70)

    for name, dataset in all_datasets.items():
        print(f"\n>>> {name}")
        print(f"    {dataset['desc']}")

        result = analyze(operator, dataset['data'], name)
        if result:
            result['description'] = dataset['desc']
            all_results.append(result)
            print(f"    Symmetry Δ: {result['symmetry_delta']:.6f}")
            print(f"    Peak/Trough: {result['peak_count']}/{result['trough_count']}")
            print(f"    Kurtosis: {result['kurtosis']:.4f}, ZCR: {result['zcr']:.6f}")
        else:
            print(f"    Insufficient data for analysis")

    # Summary
    if all_results:
        print("\n" + "=" * 70)
        print("RAW DATA RESULTS SUMMARY")
        print("=" * 70)

        print("\nRanking by Symmetry (lowest Δ = most symmetric):")
        sorted_results = sorted(all_results, key=lambda x: x['symmetry_delta'])
        for i, r in enumerate(sorted_results, 1):
            print(f"  {i:2d}. {r['signal_name']:40s}: Δ = {r['symmetry_delta']:.6f} (n={r['n_samples']})")

        # Compare to synthetic baseline
        print("\n" + "-" * 70)
        print("COMPARISON: Raw Data vs Synthetic (from previous tests)")
        print("-" * 70)
        print("\nPrevious synthetic results:")
        print("  GW events (synthetic):     Δ = 0.027 ± 0.015")
        print("  Seismic (synthetic):       Δ = 0.026 ± 0.013")
        print("  Pure noise (synthetic):    Δ = 0.023 ± 0.017")

        raw_syms = [r['symmetry_delta'] for r in all_results]
        print(f"\nRaw data results:")
        print(f"  All raw data:              Δ = {np.mean(raw_syms):.3f} ± {np.std(raw_syms):.3f}")
        print(f"  Range:                     [{np.min(raw_syms):.3f}, {np.max(raw_syms):.3f}]")

        # Save results
        results_dir = '/home/user/TEAPOT/results/raw_data_analysis'
        os.makedirs(results_dir, exist_ok=True)

        output_file = os.path.join(results_dir, 'raw_data_tensor_results.json')
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'methodology': 'RAW_UNFILTERED_DATA',
                'sources': ['USGS', 'NOAA_NDBC', 'NOAA_SWPC', 'FRED', 'CoinDesk'],
                'results': all_results
            }, f, indent=2)

        print(f"\n  Results saved to: {output_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    run_raw_data_analysis()
