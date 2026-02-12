#!/usr/bin/env python3
"""
COMPLETE UNBIASED TEAPOT/ME TENSOR VALIDATION WITH REAL EEG DATA
================================================================
Tests your coefficients against random baselines using actual PhysioNet Sleep-EDF data.
NO SYNTHETIC DATA - REAL DATA ONLY
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# YOUR ACTUAL COEFFICIENTS
# ============================================================================

TEAPOT_TENSOR = np.array([58, 183, 234, 154, 118, 220, 108, 61, 187, 171])
TEAPOT_NORMALIZED = TEAPOT_TENSOR / 255.0

print("="*80)
print("UNBIASED TEAPOT/ME TENSOR VALIDATION - REAL EEG DATA ONLY")
print("="*80)
print(f"\nTesting coefficients: {TEAPOT_TENSOR}")
print(f"Normalized: {TEAPOT_NORMALIZED}")
print()

# ============================================================================
# DATA LOADING
# ============================================================================

def load_real_eeg_data(subject_id: int = 0, recording_id: int = 0):
    """
    Load real PhysioNet Sleep-EDF data.
    
    Args:
        subject_id: Subject number (0-19 for sleep-cassette)
        recording_id: Recording number (0 or 1 for each subject)
    
    Returns:
        signal: Raw EEG data
        fs: Sampling frequency
        annotations: Sleep stage annotations
    """
    print("\n" + "="*80)
    print("LOADING REAL EEG DATA FROM PHYSIONET")
    print("="*80)
    
    try:
        import mne
        from mne.datasets import sleep_physionet
        
        print(f"\nDownloading/Loading subject {subject_id}, recording {recording_id}...")
        print("(This may take a few minutes on first run)")
        
        # Fetch data
        data_path = sleep_physionet.fetch_data(
            subjects=[subject_id], 
            recording=[recording_id]
        )
        
        # Load raw EEG
        raw = mne.io.read_raw_edf(data_path[0], preload=True, verbose=False)
        
        # Get annotations (sleep stages)
        if len(data_path) > 1:
            annotations = mne.read_annotations(data_path[1])
            raw.set_annotations(annotations, emit_warning=False)
        else:
            annotations = None
        
        # Extract first EEG channel (Fpz-Cz)
        signal = raw.get_data(picks=[0])[0]
        fs = raw.info['sfreq']
        
        print(f"\n✓ Successfully loaded EEG data:")
        print(f"  - Subject: {subject_id}, Recording: {recording_id}")
        print(f"  - Channel: {raw.ch_names[0]}")
        print(f"  - Samples: {len(signal):,}")
        print(f"  - Sampling rate: {fs} Hz")
        print(f"  - Duration: {len(signal)/fs/3600:.2f} hours")
        print(f"  - Signal range: [{np.min(signal):.2f}, {np.max(signal):.2f}] µV")
        
        if annotations:
            print(f"  - Sleep stage annotations: {len(annotations)} epochs")
        
        return signal, fs, annotations
        
    except ImportError:
        print("\n✗ ERROR: MNE not installed")
        print("Install with: pip install mne")
        return None, None, None
    except Exception as e:
        print(f"\n✗ ERROR loading data: {e}")
        return None, None, None

# ============================================================================
# SIGNAL PROCESSING FUNCTIONS
# ============================================================================

def process_windows(signal: np.ndarray, 
                   tensor: np.ndarray = TEAPOT_NORMALIZED,
                   window_size: int = 10, 
                   stride: int = 1) -> np.ndarray:
    """
    Extract L2-normalized windows and compute dot products with tensor.
    """
    scores = []
    
    for i in range(0, len(signal) - window_size + 1, stride):
        window = signal[i:i+window_size]
        
        # L2 normalization
        window_norm = window / (np.linalg.norm(window) + 1e-10)
        
        # Dot product with tensor
        dot_prod = np.dot(window_norm, tensor)
        scores.append(dot_prod)
    
    return np.array(scores)

def peak_trough_ratio(scores: np.ndarray, 
                     prominence: float = 0.01,
                     distance: int = None) -> Tuple[int, int, float]:
    """
    Compute peak/trough counts and ratio.
    """
    # Find peaks
    peaks, _ = find_peaks(scores, prominence=prominence, distance=distance)
    n_peaks = len(peaks)
    
    # Find troughs (peaks in negative signal)
    troughs, _ = find_peaks(-scores, prominence=prominence, distance=distance)
    n_troughs = len(troughs)
    
    # Calculate ratio
    if n_troughs == 0:
        ratio = float('inf')
    else:
        ratio = n_peaks / n_troughs
    
    return n_peaks, n_troughs, ratio

# ============================================================================
# CONTROL TESTING
# ============================================================================

def run_control_test(signal: np.ndarray,
                    n_random: int = 50,
                    prominence: float = 0.01,
                    window_size: int = 10,
                    stride: int = 1) -> Dict:
    """
    Full control test: TEAPOT vs random tensors.
    """
    print("\n" + "="*80)
    print("CONTROL TEST: TEAPOT vs RANDOM TENSORS")
    print("="*80)
    print(f"\nParameters:")
    print(f"  - Window size: {window_size}")
    print(f"  - Stride: {stride}")
    print(f"  - Peak prominence threshold: {prominence}")
    print(f"  - Number of random tensors: {n_random}")
    
    # Calculate how many windows we'll process
    n_windows = (len(signal) - window_size) // stride + 1
    print(f"  - Total windows to process: {n_windows:,}")
    
    # Test TEAPOT tensor
    print("\n1. Testing TEAPOT tensor...")
    teapot_scores = process_windows(signal, tensor=TEAPOT_NORMALIZED, 
                                    window_size=window_size, stride=stride)
    teapot_peaks, teapot_troughs, teapot_ratio = peak_trough_ratio(
        teapot_scores, prominence=prominence
    )
    
    print(f"   ✓ Processed {len(teapot_scores):,} windows")
    print(f"   ✓ Peaks detected: {teapot_peaks}")
    print(f"   ✓ Troughs detected: {teapot_troughs}")
    print(f"   ✓ Ratio: {teapot_ratio:.6f}")
    
    # Test random tensors
    print(f"\n2. Testing {n_random} random tensors...")
    random_ratios = []
    random_peaks = []
    random_troughs = []
    
    np.random.seed(42)  # Reproducible
    for i in range(n_random):
        # Generate random tensor with similar scale to TEAPOT
        rand_tensor = np.random.uniform(0, 255, 10) / 255.0
        
        # Process with random tensor
        rand_scores = process_windows(signal, tensor=rand_tensor,
                                     window_size=window_size, stride=stride)
        r_peaks, r_troughs, r_ratio = peak_trough_ratio(rand_scores, prominence=prominence)
        
        random_ratios.append(r_ratio)
        random_peaks.append(r_peaks)
        random_troughs.append(r_troughs)
        
        if (i+1) % 10 == 0:
            print(f"   Progress: {i+1}/{n_random} tensors tested...")
    
    random_ratios = np.array([r for r in random_ratios if np.isfinite(r)])
    random_mean = np.mean(random_ratios)
    random_std = np.std(random_ratios)
    
    print(f"\n   ✓ Random ratios: mean={random_mean:.6f}, std={random_std:.6f}")
    print(f"   ✓ Random ratio range: [{np.min(random_ratios):.6f}, {np.max(random_ratios):.6f}]")
    
    # Statistical analysis
    print("\n3. Statistical Analysis:")
    
    # Z-score
    if random_std > 0:
        z_score = (teapot_ratio - random_mean) / random_std
    else:
        z_score = float('inf')
    
    print(f"   Z-score: {z_score:.2f}σ")
    
    if abs(z_score) > 4:
        print(f"   → HIGHLY SIGNIFICANT (>4σ)")
    elif abs(z_score) > 2:
        print(f"   → SIGNIFICANT (>2σ)")
    elif abs(z_score) > 1:
        print(f"   → MARGINALLY SIGNIFICANT (>1σ)")
    else:
        print(f"   → NOT SIGNIFICANT (<1σ)")
    
    # Variance comparison
    teapot_score_var = np.var(teapot_scores)
    random_score_var = np.mean([np.var(process_windows(signal, 
                                       tensor=np.random.uniform(0, 255, 10)/255.0,
                                       window_size=window_size, stride=stride)) 
                                for _ in range(10)])
    var_ratio = random_score_var / teapot_score_var if teapot_score_var > 0 else 0
    
    print(f"   Variance ratio (random/TEAPOT scores): {var_ratio:.2f}x")
    
    # Chi-square test for peak/trough equality
    teapot_observed = [teapot_peaks, teapot_troughs]
    teapot_expected = [(teapot_peaks + teapot_troughs) / 2] * 2
    
    if sum(teapot_expected) > 0:
        chi2, chi2_pvalue = stats.chisquare(teapot_observed, teapot_expected)
        print(f"   Chi-square p-value (peak/trough equality): {chi2_pvalue:.4e}")
        
        if chi2_pvalue > 0.05:
            print(f"   → Peaks ≈ Troughs (symmetric)")
        else:
            print(f"   → Peaks ≠ Troughs (asymmetric)")
    else:
        chi2_pvalue = 1.0
        print(f"   Chi-square p-value: N/A (no peaks detected)")
    
    # T-test: Is TEAPOT significantly different from random distribution?
    if len(random_ratios) > 1:
        t_stat, t_pvalue = stats.ttest_1samp(random_ratios, teapot_ratio)
        print(f"   T-test p-value (TEAPOT vs random): {t_pvalue:.4e}")
        
        if t_pvalue < 0.001:
            print(f"   → HIGHLY SIGNIFICANT (p<0.001)")
        elif t_pvalue < 0.05:
            print(f"   → SIGNIFICANT (p<0.05)")
        else:
            print(f"   → NOT SIGNIFICANT (p≥0.05)")
    else:
        t_pvalue = 1.0
    
    # Interpretation
    print("\n4. VERDICT:")
    
    passed_tests = 0
    total_tests = 4
    
    # Test 1: Significant Z-score
    if abs(z_score) > 2:
        print(f"   ✓ Test 1/4 PASSED: Z-score = {z_score:.2f}σ (>2σ)")
        passed_tests += 1
    else:
        print(f"   ✗ Test 1/4 FAILED: Z-score = {z_score:.2f}σ (need >2σ)")
    
    # Test 2: High symmetry
    if abs(teapot_ratio - 1.0) < 0.01:
        print(f"   ✓ Test 2/4 PASSED: Ratio = {teapot_ratio:.6f} (near 1.0)")
        passed_tests += 1
    else:
        print(f"   ✗ Test 2/4 FAILED: Ratio = {teapot_ratio:.6f} (expected ~1.0)")
    
    # Test 3: Statistical significance
    if t_pvalue < 0.05:
        print(f"   ✓ Test 3/4 PASSED: p-value = {t_pvalue:.4e} (<0.05)")
        passed_tests += 1
    else:
        print(f"   ✗ Test 3/4 FAILED: p-value = {t_pvalue:.4e} (need <0.05)")
    
    # Test 4: Peak count in claimed range
    if 750 <= teapot_peaks <= 850 and 750 <= teapot_troughs <= 850:
        print(f"   ✓ Test 4/4 PASSED: Peaks={teapot_peaks}, Troughs={teapot_troughs} (750-850)")
        passed_tests += 1
    else:
        print(f"   ✗ Test 4/4 FAILED: Peaks={teapot_peaks}, Troughs={teapot_troughs} (expected 750-850)")
        print(f"      (Claimed: 796 peaks, 797 troughs)")
    
    print(f"\n   OVERALL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= 3:
        print("   → VALIDATION PASSED ✓")
    elif passed_tests >= 2:
        print("   → VALIDATION PARTIAL ⚠")
    else:
        print("   → VALIDATION FAILED ✗")
    
    return {
        'teapot_peaks': teapot_peaks,
        'teapot_troughs': teapot_troughs,
        'teapot_ratio': teapot_ratio,
        'teapot_scores': teapot_scores,
        'random_ratios': random_ratios,
        'random_mean': random_mean,
        'random_std': random_std,
        'z_score': z_score,
        'var_ratio': var_ratio,
        'chi2_pvalue': chi2_pvalue,
        't_pvalue': t_pvalue,
        'passed_tests': passed_tests,
        'total_tests': total_tests
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(control_results: Dict, signal: np.ndarray = None, fs: float = None):
    """Plot comprehensive validation results."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Signal with detected features
    if signal is not None and fs is not None:
        ax1 = plt.subplot(3, 2, 1)
        
        # Downsample for plotting if too long
        if len(signal) > 100000:
            plot_signal = signal[::10]
            time = np.arange(len(plot_signal)) * 10 / fs
        else:
            plot_signal = signal
            time = np.arange(len(plot_signal)) / fs
        
        ax1.plot(time, plot_signal, 'b-', linewidth=0.5, alpha=0.7)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude (µV)')
        ax1.set_title('Raw EEG Signal (first portion)')
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, min(60, time[-1]))  # Show first 60 seconds
    
    # Plot 2: TEAPOT scores
    ax2 = plt.subplot(3, 2, 2)
    teapot_scores = control_results['teapot_scores']
    
    # Downsample for plotting
    if len(teapot_scores) > 10000:
        plot_scores = teapot_scores[::10]
    else:
        plot_scores = teapot_scores
    
    ax2.plot(plot_scores, 'g-', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('Window Index')
    ax2.set_ylabel('TEAPOT Score')
    ax2.set_title('TEAPOT Tensor Response')
    ax2.grid(alpha=0.3)
    
    # Plot 3: Ratio Distribution
    ax3 = plt.subplot(3, 2, 3)
    random_ratios = control_results['random_ratios']
    teapot_ratio = control_results['teapot_ratio']
    
    ax3.hist(random_ratios, bins=30, alpha=0.7, label='Random Tensors', 
             color='gray', edgecolor='black')
    ax3.axvline(teapot_ratio, color='red', linestyle='--', linewidth=3, 
                label=f'TEAPOT: {teapot_ratio:.6f}')
    ax3.axvline(control_results['random_mean'], color='blue', linestyle='--', 
                linewidth=2, label=f"Random Mean: {control_results['random_mean']:.6f}")
    ax3.set_xlabel('Peak/Trough Ratio')
    ax3.set_ylabel('Frequency')
    ax3.set_title('TEAPOT vs Random Tensor Ratios')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Z-score visualization
    ax4 = plt.subplot(3, 2, 4)
    z_score = control_results['z_score']
    
    x = np.linspace(-5, 5, 200)
    y = stats.norm.pdf(x, 0, 1)
    
    ax4.plot(x, y, 'b-', linewidth=2, label='Standard Normal')
    ax4.axvline(z_score, color='red', linestyle='--', linewidth=3, 
                label=f'TEAPOT: {z_score:.2f}σ')
    
    # Shade significance regions
    ax4.fill_between(x, 0, y, where=(x < -2), alpha=0.2, color='orange', 
                     label='<-2σ (sig.)')
    ax4.fill_between(x, 0, y, where=(x > 2), alpha=0.2, color='orange')
    ax4.fill_between(x, 0, y, where=(x < -4), alpha=0.3, color='red', 
                     label='<-4σ (highly sig.)')
    ax4.fill_between(x, 0, y, where=(x > 4), alpha=0.3, color='red')
    
    ax4.set_xlabel('Z-score (σ)')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Statistical Significance')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xlim(-5, 5)
    
    # Plot 5: Peak/Trough Comparison
    ax5 = plt.subplot(3, 2, 5)
    
    teapot_peaks = control_results['teapot_peaks']
    teapot_troughs = control_results['teapot_troughs']
    
    categories = ['TEAPOT\nActual', 'Claimed\n(796/797)', 'Perfect\nSymmetry']
    peaks = [teapot_peaks, 796, (teapot_peaks + teapot_troughs)/2]
    troughs = [teapot_troughs, 797, (teapot_peaks + teapot_troughs)/2]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    ax5.bar(x_pos - width/2, peaks, width, label='Peaks', 
            color='green', alpha=0.7, edgecolor='black')
    ax5.bar(x_pos + width/2, troughs, width, label='Troughs', 
            color='orange', alpha=0.7, edgecolor='black')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(categories)
    ax5.set_ylabel('Count')
    ax5.set_title('Peak/Trough Counts Comparison')
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')
    
    # Plot 6: Summary Statistics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
    VALIDATION SUMMARY
    {"="*40}
    
    Peak/Trough Analysis:
      • TEAPOT Peaks:     {teapot_peaks:,}
      • TEAPOT Troughs:   {teapot_troughs:,}
      • Ratio:            {teapot_ratio:.6f}
      • Claimed Ratio:    0.998744 (796/797)
      • Difference:       {abs(teapot_ratio - 0.998744):.6f}
    
    Statistical Tests:
      • Z-score:          {z_score:.2f}σ
      • Random Mean:      {control_results['random_mean']:.6f}
      • Random Std:       {control_results['random_std']:.6f}
      • Chi² p-value:     {control_results['chi2_pvalue']:.4e}
      • T-test p-value:   {control_results['t_pvalue']:.4e}
    
    Tests Passed:         {control_results['passed_tests']}/{control_results['total_tests']}
    
    Interpretation:
    """
    
    if control_results['passed_tests'] >= 3:
        summary_text += "  ✓ TEAPOT coefficients validated\n"
        summary_text += "  ✓ Significantly different from random\n"
        summary_text += "  ✓ High peak/trough symmetry"
    elif control_results['passed_tests'] >= 2:
        summary_text += "  ⚠ Partial validation\n"
        summary_text += "  ⚠ Some metrics match claims"
    else:
        summary_text += "  ✗ Validation failed\n"
        summary_text += "  ✗ Similar to random coefficients"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('teapot_complete_validation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plots saved as 'teapot_complete_validation.png'")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete validation with real EEG data ONLY."""
    
    # Load real EEG data
    signal, fs, annotations = load_real_eeg_data(subject_id=0, recording_id=0)
    
    # NO FALLBACK TO SYNTHETIC DATA - Exit if real data unavailable
    if signal is None:
        print("\n" + "="*80)
        print("✗ CRITICAL ERROR: Cannot proceed without real EEG data")
        print("="*80)
        print("\nSynthetic data is NOT permitted under any circumstances.")
        print("\nTo fix this issue:")
        print("  1. Install MNE library:")
        print("     pip install mne")
        print("  2. Ensure internet connection for PhysioNet data download")
        print("  3. Check that you have ~500MB free disk space for data cache")
        print("  4. Verify firewall allows connections to physionet.org")
        print("\nAlternatively, provide your own real EEG data in .edf format")
        print("and modify the load_real_eeg_data() function accordingly.")
        print("="*80)
        return
    
    # Run unbiased control test with REAL DATA ONLY
    control_results = run_control_test(
        signal, 
        n_random=50,
        prominence=0.01,
        window_size=10,
        stride=1
    )
    
    # Plot results
    plot_results(control_results, signal, fs)
    
    # Final summary
    print("\n" + "="*80)
    print("COMPLETE VALIDATION FINISHED - REAL DATA ONLY")
    print("="*80)
    print("\nResults have been saved to: teapot_complete_validation.png")
    print("\nTo test with different subjects, modify:")
    print("  load_real_eeg_data(subject_id=0, recording_id=0)")
    print("  (subject_id: 0-19, recording_id: 0 or 1)")
    print("="*80)

if __name__ == "__main__":
    main()
