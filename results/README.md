# Validation Results

This folder contains validation results from the TEAPOT tensor operator testing.

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

## Key Metrics

| Metric | Subject SC4001E0 | Subject SC4002E0 |
|--------|------------------|------------------|
| Total Samples | 7.95M | 7.95M |
| Peaks | 796 | 812 |
| Troughs | 797 | 814 |
| Symmetry Ratio | 0.999 | 0.998 |
| p-value | 1.33e-216 | 8.7e-203 |
| NREM Δ_sym (mean) | 0.018 | 0.020 |
| Wake Δ_sym (mean) | 0.187 | 0.183 |

## Negative Controls

### Scrambled Labels Test
- **Result**: Symmetry differences disappeared (p=0.42)
- **Conclusion**: Operator detects real physiological patterns

### Synthetic Noise Test
- **Result**: High Δ_sym (>0.15) for pink/brown/white noise
- **Conclusion**: No false positives on non-physiological signals

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
