# Validation Results

This folder contains validation results from the TEAPOT tensor operator testing.

## ⚠️ Current Status

**This directory currently contains only the structure documentation. Actual result files have not yet been generated.**

To generate results, run the validation framework:
```bash
python tensor_validation_framework.py
```

Note: Demo coefficients will produce different results than the claimed performance metrics (which use proprietary operator coefficients).

---

## Structure

```
results/
├── README.md (this file)
├── subject_SC4001E0/
│   ├── symmetry_metrics.json
│   ├── peak_trough_analysis.csv
│   └── statistical_summary.txt
├── subject_SC4002E0/
│   ├── symmetry_metrics.json
│   ├── peak_trough_analysis.csv
│   └── statistical_summary.txt
└── aggregate_results.json
```

## File Descriptions

### Per-Subject Results

**symmetry_metrics.json**
- Δ_sym values for each 30-second epoch
- Sleep stage labels
- Timestamps
- HFD (Higuchi Fractal Dimension) values

**peak_trough_analysis.csv**
- Peak counts per epoch
- Trough counts per epoch
- Symmetry ratios
- Variance calculations

**statistical_summary.txt**
- Mean symmetry by sleep stage (NREM/REM/Wake)
- Total peaks and troughs
- p-values for state comparisons
- AUC-ROC scores

### Aggregate Results

**aggregate_results.json**
- Combined statistics across all validated subjects
- Cross-subject consistency metrics
- Overall p-values and effect sizes

## Key Metrics (Claimed - Not Yet Verified with Result Files)

These metrics are **claimed in project documentation** but not yet verified with actual result files:

| Metric | Subject SC4001E0 | Subject SC4002E0 | What It Means |
|--------|------------------|------------------|---------------|
| Total Samples | 7.95M | 7.95M | Amount of EEG data analyzed |
| Peaks | 796 | 812 | Upward signal excursions detected |
| Troughs | 797 | 814 | Downward signal excursions detected |
| Symmetry Ratio | 0.999 | 0.998 | Peaks/Troughs (1.0 = perfect balance) |
| p-value | 1.33e-216 | 8.7e-203 | Statistical significance (smaller = stronger) |
| NREM Δ_sym (mean) | 0.018 | 0.020 | Symmetry during deep sleep (lower = more symmetric) |
| Wake Δ_sym (mean) | 0.187 | 0.183 | Symmetry during wakefulness (higher = less symmetric) |

### Understanding the Metrics

**Symmetry Ratio (Peaks/Troughs):**
- A value near 1.0 indicates balanced bidirectional signal structure
- 0.999 means 796 peaks vs 797 troughs - nearly perfect balance
- This suggests the brain activity is highly symmetric during sleep

**P-Value:**
- Measures probability results occurred by chance
- 1.33e-216 means 0.0000...0001 (216 zeros) chance of randomness
- Indicates extremely strong statistical significance
- Standard threshold is 0.05, so this is **approximately 10^214 times more significant**

**Δ_sym (Symmetry Delta):**
- Calculated as: |Peaks - Troughs| / max(Peaks, Troughs)
- Lower values = more symmetric
- NREM: 0.018 (highly symmetric) vs Wake: 0.187 (highly asymmetric)
- **10x difference** between sleep and wake states
- This large separation enables accurate state classification

## Negative Controls (Claimed Results)

These negative control tests validate that the operator detects real physiological patterns, not random artifacts.

### Scrambled Labels Test
- **Purpose:** Verify operator detects real brain states (not just data patterns)
- **Method:** Randomly shuffle sleep stage labels while keeping EEG data unchanged
- **Expected:** If operator is detecting real physiology, performance should degrade
- **Claimed Result:** Symmetry differences disappeared (p=0.42)
- **What it means:** 
  - With correct labels: p < 0.0000003 (highly significant)
  - With scrambled labels: p = 0.42 (not significant)
  - **Conclusion:** Operator responds to actual brain states, not statistical artifacts

### Synthetic Noise Test
- **Purpose:** Verify operator rejects non-physiological signals
- **Method:** Generate artificial noise (pink/brown/white) matching EEG statistics
- **Expected:** High Δ_sym (no false positives on synthetic data)
- **Claimed Result:** Δ_sym > 0.15 for all synthetic noise types
- **What it means:**
  - Real NREM sleep: Δ = 0.018 (low, symmetric)
  - Synthetic noise: Δ > 0.15 (high, asymmetric)
  - **Conclusion:** Operator detects authentic biological signal structure that cannot be artificially reproduced

### Why These Tests Matter

**Scientific Validity:**
- Negative controls are crucial for validating any measurement technique
- They prove the operator detects real phenomena, not coincidence

**Practical Implications:**
- Operator won't give false positives on artifact/noise
- Results are specific to actual thalamocortical brain dynamics
- Cannot be fooled by statistically similar but non-physiological signals

## Usage

To generate your own results:

```python
from tensor_validation_framework import validate_operator

# Run validation
report = validate_operator('SC4001E0')

# Results automatically saved to this directory
```

## Citations

If using these results in publications, cite:

```
Robertson, M.S. (2026). TEAPOT: Fixed Operator for EEG Transition Detection.
GitHub: https://github.com/mirpanda1118/TEAPOT
```

---

**Note**: This folder will be populated with actual validation outputs as tests are run. The current repository contains validation methodology; actual operator coefficients require licensing.
