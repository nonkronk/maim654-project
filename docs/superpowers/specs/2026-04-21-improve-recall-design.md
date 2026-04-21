# Improve Recall for Electricity Theft Detection

## Problem Statement

Current model achieves **61% recall** on the test set — nearly 40% of theft events go undetected. For electricity theft detection, false negatives (missing actual theft) represent ongoing revenue loss that compounds over time, while false positives only trigger investigation costs. Recall is the primary success metric.

Baseline results:
- Precision: 0.9422
- **Recall: 0.6112**
- F1-Score: 0.7415
- Balanced Accuracy: 0.8035
- ROC-AUC: 0.8718
- PR-AUC: 0.7443

## Scope

Three orthogonal improvements targeting the data pipeline, feature engineering, and model training. All changes stay within the existing notebook (`colab/notebook.ipynb`) — no new files or external dependencies.

---

## Section 1: Stronger Synthetic Theft Patterns

### Current State

`inject_theft()` (notebook cell 5) implements 5 mild attack patterns:
1. Scale by 0.25-0.75x
2. Random spike profile with 1.5-3.0x strength
3. Constant flat-line
4. Random mask (45% zeros)
5. Rolling shift by 3-8 positions

These patterns are too subtle — mild scaling and brief spikes resemble natural load variation, giving the model easy discrimination rather than learnable theft signatures.

### Changes

Replace `inject_theft()` with 9 patterns. Keep all 5 existing patterns; add 4 new ones:

| # | Pattern Name | Description | Anomaly Budget Share |
|---|---|---|---|
| 0 | Partial bypass ramp (NEW) | Linearly decrease consumption from 100% to 20% over the window | ~11% |
| 1 | Spike (existing) | Random spike profile with increased strength | ~7% |
| 2 | Flat-line (existing) | Set all values to a small constant | ~7% |
| 3 | Intermittent tampering (NEW) | Zero-out random sub-segments with realistic gaps between them | ~11% |
| 4 | Progressive decay (NEW) | Exponential decay from normal to near-zero mimicking slow meter degradation | ~11% |
| 5 | Pulse theft (NEW) | Subtract baseline during specific sub-windows | ~11% |
| 6 | Scale (existing) | Uniform scale by 0.25-0.75x | ~7% |
| 7 | Mask (existing) | Random binary mask | ~7% |
| 8 | Shift (existing) | Rolling shift with fill | ~7% |

Total anomaly rate stays at 10%. Each pattern type gets roughly equal representation in the synthetic theft budget.

### Implementation

Modify the `inject_theft()` function and its caller `generate_synthetic_labels()` in notebook cell 5/6. No structural changes to the pipeline — same data flow, just different label injection logic.

---

## Section 2: Richer Feature Engineering

### Current State

`build_window_features()` (notebook cell 5) produces 14 features per time step:
- load (raw values)
- hour_sin, hour_cos (temporal cyclical encoding)
- win_mean, win_std, win_max, win_min (aggregate statistics)
- load_ratio (normalized to window mean)
- flatness, mean_abs_diff, diff_std, zero_fraction, slope, range_ratio (temporal dynamics)

All features are time-domain. No frequency information or multi-scale statistics.

### Changes

Add three new feature groups (11 additional features per time step):

#### FFT Spectral Features (4 features)
- Low-freq energy: ratio of spectral power in 0-3 Hz band
- Mid-freq energy: ratio of spectral power in 3-12 Hz band
- High-freq entropy: Shannon entropy of the normalized power spectrum
- Spectral centroid: weighted center of mass of the spectrum

#### Multi-scale Rolling Statistics (4 features)
- Mean/std over rolling windows of size 6 and 12
- Ratio of current window mean to each rolling window mean (captures deviation from local baseline)

#### Temporal Autocorrelation (3 features)
- Lag-1, lag-3, lag-6 autocorrelation coefficients
- Hanning-smoothed series reconstruction error

**Total feature dimension:** 14 → 25 per time step.

### Implementation

Extend `build_window_features()` in notebook cell 5. Stack new feature arrays alongside existing ones along the feature axis. The LSTM input_size parameter handles the change automatically.

---

## Section 3: Asymmetric Loss + Training Tweaks

### Current State

- Loss: FocalLoss(alpha=0.8, gamma=2.0) — class-weighted but symmetric on false positives/negatives
- Learning rate: 3e-4 with ReduceLROnPlateau
- Threshold optimization: maximize F1-score
- Training: WeightedRandomSampler with inverse class frequency

### Changes

#### New Combined Loss Function

Replace `FocalLoss` with `TverskyFocalLoss`:

```python
class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, lambda_focal=0.5):
        # alpha > beta means false negatives weighted higher in Tversky component
        # lambda_focal controls the focal loss contribution

    def forward(self, logits, targets):
        tversky = _tversky_loss(logits, targets, alpha, beta)
        focal = _focal_loss_component(logits, targets, gamma)
        return tversky + lambda_focal * focal
```

Parameters: alpha=0.7, beta=0.3 (FN penalized ~2.3x more than FP), gamma=2.0, lambda_focal=0.5.

#### Training Hyperparameter Changes

| Parameter | Before | After | Rationale |
|---|---|---|---|
| Focal alpha component | 0.8 | 0.9 | Heavier theft weighting |
| Base LR | 3e-4 | 1e-4 | Finer convergence near recall-optimal region |

#### Decision Threshold Strategy

Replace F1-maximize threshold selection with **recall-target constraint**:
- Find the lowest threshold that achieves **≥80% validation recall**
- Constraint: precision must remain **≥0.85** (avoid excessive false alarms)
- Select the threshold meeting both criteria closest to the minimum

### Implementation

Add `TverskyFocalLoss` class in cell 10, update criterion initialization, and modify `optimize_threshold()` to use the recall-target constraint instead of F1-maximization.

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Feature explosion causing overfitting | L2 regularization already in optimizer (weight_decay=1e-4); increase if val loss diverges |
| Tversky loss instability | Clamp probabilities to [1e-6, 1-1e-6]; add gradient clipping (already present at norm=1.0) |
| More theft patterns reducing signal clarity | Equal allocation per pattern prevents any single pattern from dominating; monitor per-pattern detection rates |
| Threshold constraint infeasible | If recall target unattainable, report the best achievable and log for further investigation |

## Success Criteria

- Target: Recall ≥ 0.80 on test set (up from 0.61)
- Precision should remain ≥ 0.85 (up from 0.94 — acceptable tradeoff if recall improves significantly)
- F1 should improve by at least 0.10 points
