# Improve Recall for Electricity Theft Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise electricity theft detection recall from 0.61 to >= 0.80 by strengthening synthetic theft patterns, adding discriminative features, and using asymmetric loss function.

**Architecture:** Three independent improvement layers applied sequentially — data injection → feature engineering → model loss/training. Each builds on the previous without requiring structural notebook changes. All modifications are in-place within `colab/notebook.ipynb`.

**Tech Stack:** Python 3, PyTorch, NumPy, scikit-learn (all existing dependencies)

---

## Task 1: Add Stronger Theft Patterns to `inject_theft`

**Files:**
- Modify: `colab/notebook.ipynb` cell 6 (replace the `inject_theft` function)

- [ ] **Step 1: Replace `inject_theft` function in cell 6**

Current cell 6 has `inject_theft` with theft_type integers 0-4. Replace it with a function that selects from 9 theft patterns (5 existing + 4 new):

```python
def inject_theft(window, rng):
    """Apply one of nine realistic synthetic theft patterns to a window."""
    window = np.asarray(window, dtype=np.float32).copy()
    theft_type = int(rng.integers(0, 9))
    abs_window = np.abs(window)
    mean_abs = max(float(np.mean(abs_window)), 1e-3)
    win_max = float(np.max(window))

    if theft_type == 0:
        # Partial bypass ramp — linear decrease from full to ~20%
        ramp = np.linspace(1.0, 0.2, len(window), dtype=np.float32)
        return (window * ramp).astype(np.float32)

    if theft_type == 1:
        # Spike — existing pattern with stronger magnitude
        spike_profile = rng.uniform(0.25, 1.0, size=window.shape).astype(np.float32)
        spike_strength = float(rng.uniform(2.0, 4.0)) * mean_abs
        return (window + spike_profile * spike_strength).astype(np.float32)

    if theft_type == 2:
        # Flat-line — existing pattern
        constant = float(rng.uniform(0.0, max(win_max * 0.05, 1e-3)))
        return np.full_like(window, constant, dtype=np.float32)

    if theft_type == 3:
        # Intermittent tampering — zero-out random sub-segments with gaps
        mask = np.ones(len(window), dtype=np.float32)
        n_segments = rng.integers(1, 4)
        for _ in range(n_segments):
            seg_start = rng.integers(0, len(window) - 3)
            seg_len = rng.integers(2, min(8, len(window) - seg_start))
            mask[seg_start:seg_start + seg_len] = 0.0
        return (window * mask).astype(np.float32)

    if theft_type == 4:
        # Progressive decay — exponential decrease toward zero
        decay_rate = rng.uniform(0.1, 0.5)
        decay = np.exp(-decay_rate * np.arange(len(window), dtype=np.float32))
        target_level = rng.uniform(0.02, 0.1)
        profile = target_level + (1.0 - target_level) * decay
        return (window * profile.astype(np.float32)).astype(np.float32)

    if theft_type == 5:
        # Pulse theft — subtract baseline during specific sub-windows
        pulse_mask = np.zeros(len(window), dtype=np.float32)
        n_pulses = rng.integers(1, 4)
        for _ in range(n_pulses):
            p_start = rng.integers(0, len(window) - 2)
            p_len = rng.integers(1, min(5, len(window) - p_start))
            pulse_mask[p_start:p_start + p_len] = 1.0
        subtraction = mean_abs * rng.uniform(0.5, 1.5)
        return (window - pulse_mask * subtraction).astype(np.float32)

    if theft_type == 6:
        # Scale — existing pattern
        scale = float(rng.uniform(0.25, 0.75))
        return (window * scale).astype(np.float32)

    if theft_type == 7:
        # Mask — existing pattern
        mask = rng.choice([0.0, 1.0], size=window.shape, p=[0.45, 0.55]).astype(np.float32)
        return (window * mask).astype(np.float32)

    if theft_type == 8:
        # Shift — existing pattern
        shift = int(rng.integers(3, min(8, len(window))))
        shifted = np.roll(window, shift).astype(np.float32)
        shifted[:shift] = shifted[shift]
        return shifted

    raise ValueError(f'Unknown theft type: {theft_type}')
```

**Step 2: Update `generate_synthetic_labels` caller in cell 6 to track per-pattern stats**

Update the existing `generate_synthetic_labels` function in cell 6 to count how many of each pattern were injected (for diagnostics). Add a global counter and update logging in cell 8. No structural changes — just append a `pattern_counts = np.zeros(9)` array, increment it inside the anomaly loop, and return it alongside the labels. Update the print statement in cell 8 to show per-pattern counts.

- [ ] **Step 3: Commit**

```bash
git add colab/notebook.ipynb
git commit -m "$(cat <<'EOF'
feat: add 4 stronger synthetic theft patterns (ramp, intermittent, decay, pulse)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add FFT Spectral Features

**Files:**
- Modify: `colab/notebook.ipynb` cell 6 (extend `build_window_features` function)

- [ ] **Step 1: Add FFT spectral feature computation to `build_window_features`**

In cell 6, add these computations after the existing features and before the final `np.stack`:

```python
    # FFT spectral features (4 new features, repeated across window length)
    fft_vals = np.fft.rfft(load_windows, axis=1)  # shape: (n_samples, window_size//2 + 1)
    power_spectrum = np.abs(fft_vals) ** 2

    n_freqs = power_spectrum.shape[1]
    low_freq_end = max(1, int(n_freqs * 0.15))
    mid_freq_start = low_freq_end
    mid_freq_end = max(low_freq_end + 1, int(n_freqs * 0.5))

    total_power = power_spectrum.sum(axis=1, keepdims=True) + 1e-8
    low_freq_energy = (power_spectrum[:, :low_freq_end].sum(axis=1, keepdims=True) / total_power).astype(np.float32)
    mid_freq_energy = (power_spectrum[:, mid_freq_start:mid_freq_end].sum(axis=1, keepdims=True) / total_power).astype(np.float32)

    # Spectral entropy
    normalized_spectrum = power_spectrum / (total_power + 1e-8)
    spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10), axis=1, keepdims=True).astype(np.float32)

    # Spectral centroid
    freq_indices = np.arange(n_freqs, dtype=np.float32)[None, :]
    spectral_centroid = (freq_indices * power_spectrum).sum(axis=1, keepdims=True) / (power_spectrum.sum(axis=1, keepdims=True) + 1e-8)

    fft_features = np.hstack([low_freq_energy, mid_freq_energy, spectral_entropy, spectral_centroid]).astype(np.float32)
    # Repeat across window length
    fft_features = np.repeat(fft_features[:, :, np.newaxis], window_size, axis=2)
```

- [ ] **Step 2: Update the `np.stack` call and `FEATURE_NAMES` list in cell 6**

Replace the existing stack call to include the 4 new FFT features. The updated feature list:

```python
    features = np.stack(
        [
            load_windows,
            hour_sin,
            hour_cos,
            win_mean,
            win_std,
            win_max,
            win_min,
            load_ratio,
            flatness,
            mean_abs_diff,
            diff_std,
            zero_fraction,
            slope,
            range_ratio,
            low_freq_energy,
            mid_freq_energy,
            spectral_entropy,
            spectral_centroid,
        ],
        axis=-1,
    )
```

And update `FEATURE_NAMES` to include:
```python
    'low_freq_energy',
    'mid_freq_energy',
    'spectral_entropy',
    'spectral_centroid',
```

- [ ] **Step 3: Commit**

```bash
git add colab/notebook.ipynb
git commit -m "$(cat <<'EOF'
feat: add FFT spectral features for frequency-domain theft detection

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add Multi-scale Rolling Stats and Autocorrelation Features

**Files:**
- Modify: `colab/notebook.ipynb` cell 6 (extend `build_window_features` function)

- [ ] **Step 1: Add multi-scale rolling stats to `build_window_features` in cell 6**

Add this after the FFT features and before the stack call:

```python
    # Multi-scale rolling statistics (4 new features)
    roll_means = []
    roll_stds = []
    roll_ratios = []
    for rs in [6, 12]:
        rolled = np.lib.stride_tricks.sliding_window_view(load_windows, window_size=rs, axis=1)
        roll_means.append(rolled.mean(axis=-1).astype(np.float32))  # shape: (n, original_len - rs + 1)
        roll_stds.append(rolled.std(axis=-1).astype(np.float32))
        ratio = load_windows[:, -(rs-1):] / (roll_means[-1][:, :, np.newaxis] + 1e-8)
        roll_ratios.append(ratio.astype(np.float32))

    # Pad shorter windows to match window_size
    def pad_to_window(arr, window_size):
        if arr.shape[1] == window_size:
            return arr
        pad_left = (window_size - arr.shape[1]) // 2
        pad_right = window_size - arr.shape[1] - pad_left
        return np.pad(arr, ((0, 0), (pad_left, pad_right)), mode='edge')

    ms_features = np.hstack([
        pad_to_window(roll_means[0], window_size),
        pad_to_window(roll_stds[0], window_size),
        pad_to_window(roll_means[1], window_size),
        pad_to_window(roll_stds[1], window_size),
    ]).astype(np.float32)[:, :, np.newaxis]

    # Temporal autocorrelation (3 new features)
    mean_val = load_windows.mean(axis=1, keepdims=True)
    centered = load_windows - mean_val
    var_all = np.sum(centered ** 2, axis=1, keepdims=True) + 1e-8

    def autocorr(lag):
        c = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1, keepdims=True) / var_all
        return c.astype(np.float32)

    ac_features = np.hstack([
        autocorr(1),
        autocorr(3),
        autocorr(6),
    ]).astype(np.float32)[:, :, np.newaxis]

    # Pad autocorrelation to match window size
    ac_padded = np.repeat(ac_features, window_size, axis=2)
```

- [ ] **Step 2: Update the `np.stack` and `FEATURE_NAMES` lists in cell 6**

Add these new features to the stack call and update FEATURE_NAMES:

```python
    features = np.stack(
        [
            load_windows,
            hour_sin,
            hour_cos,
            win_mean,
            win_std,
            win_max,
            win_min,
            load_ratio,
            flatness,
            mean_abs_diff,
            diff_std,
            zero_fraction,
            slope,
            range_ratio,
            low_freq_energy,
            mid_freq_energy,
            spectral_entropy,
            spectral_centroid,
            ms_features,  # multi-scale rolling stats (4 features)
            ac_padded,    # autocorrelation (3 features)
        ],
        axis=-1,
    )
```

Updated FEATURE_NAMES:
```python
FEATURE_NAMES = [
    'load', 'hour_sin', 'hour_cos', 'win_mean', 'win_std', 'win_max', 'win_min',
    'load_ratio', 'flatness', 'mean_abs_diff', 'diff_std', 'zero_fraction',
    'slope', 'range_ratio',
    'low_freq_energy', 'mid_freq_energy', 'spectral_entropy', 'spectral_centroid',
    'roll_mean_6', 'roll_std_6', 'roll_mean_12', 'roll_std_12',
    'autocorr_lag1', 'autocorr_lag3', 'autocorr_lag6',
]
```

- [ ] **Step 3: Commit**

```bash
git add colab/notebook.ipynb
git commit -m "$(cat <<'EOF'
feat: add multi-scale rolling stats and autocorrelation features

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement TverskyFocalLoss Asymmetric Loss

**Files:**
- Modify: `colab/notebook.ipynb` cell 11 (replace/add loss functions)

- [ ] **Step 1: Add `TverskyFocalLoss` class to cell 11, before existing loss classes**

Insert this class right after the imports and before `class FocalLoss`:

```python
class TverskyFocalLoss(nn.Module):
    """Combined Tversky + Focal loss that heavily penalizes false negatives."""
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, lambda_focal=0.5):
        super().__init__()
        self.alpha = alpha  # FP weight in Tversky (lower = more FN emphasis)
        self.beta = beta     # FN weight in Tversky (higher = more FN penalty)
        self.gamma = gamma   # Focal modulating factor
        self.lambda_focal = lambda_focal

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        probs_clamped = torch.clamp(probs, 1e-6, 1.0 - 1e-6)

        tp = (probs_clamped * targets).sum(dim=0)
        fp = (probs_clamped * (1.0 - targets)).sum(dim=0)
        fn = ((1.0 - probs_clamped) * targets).sum(dim=0)

        tversky_num = tp + 1e-8
        tversky_den = tp + self.alpha * fp + self.beta * fn + 1e-8
        tversky = tversky_num / tversky_den
        tversky_loss = 1.0 - tversky

        # Focal component on top
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.where(targets == 1.0, probs_clamped, 1.0 - probs_clamped)
        focal_mod = torch.pow(1.0 - pt, self.gamma)
        alpha_t = torch.where(targets == 1.0,
                              torch.full_like(targets, 0.9),
                              torch.full_like(targets, 0.1))
        focal_loss = alpha_t * focal_mod * bce

        return (tversky_loss + self.lambda_focal * focal_loss).mean()
```

- [ ] **Step 2: Update criterion initialization in cell 11**

Find the existing `criterion = FocalLoss(...)` line and replace it. Also add a new variable to hold both losses:

Replace this exact block (around line ~30 of cell 11):
```python
criterion = FocalLoss(alpha=0.8, gamma=1.5)
```

With:
```python
# Primary loss: asymmetric Tversky + Focal combination
tversky_criterion = TverskyFocalLoss(alpha=0.7, beta=0.3, gamma=2.0, lambda_focal=0.5)
criterion = tversky_criterion  # kept for backward compat with total_objective_loss
```

- [ ] **Step 3: Update learning rate in cell 11**

Find `adamw_kwargs = {'lr': 2.5e-4, 'weight_decay': 1e-4}` and change to:
```python
adamw_kwargs = {'lr': 1e-4, 'weight_decay': 1e-4}
```

- [ ] **Step 4: Update recall target and precision floor**

Change `TARGET_RECALL = 0.70` to `TARGET_RECALL = 0.80` and change `MIN_PRECISION_FLOOR = 0.35` to `MIN_PRECISION_FLOOR = 0.85`.

- [ ] **Step 5: Commit**

```bash
git add colab/notebook.ipynb
git commit -m "$(cat <<'EOF'
feat: add TverskyFocalLoss asymmetric loss and raise recall target to 0.80

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update Threshold Selection for Recall-Target Constraint

**Files:**
- Modify: `colab/notebook.ipynb` cell 11 (update `optimize_threshold`)

The current `optimize_threshold` function uses `max(all_metrics, key=lambda item: (item['f1'], item['f2'], ...))` which prioritizes F1. Replace it with a recall-target constraint approach.

- [ ] **Step 1: Rewrite `optimize_threshold` function in cell 11**

Replace the existing `optimize_threshold` function body. Keep the signature and threshold_rows logic but change the final selection from "best by F1" to "best recall >= target with precision floor":

```python
def optimize_threshold(y_true_array, prob_array, thresholds=threshold_grid):
    """Select threshold that meets recall target while maintaining minimum precision."""
    threshold_rows = []
    all_metrics = []

    for threshold_value in thresholds:
        predictions = (prob_array >= threshold_value).astype(np.int64)
        precision_value = precision_score(y_true_array, predictions, zero_division=0)
        recall_value = recall_score(y_true_array, predictions, zero_division=0)
        f1_value = f1_score(y_true_array, predictions, zero_division=0)
        f2_value = safe_fbeta(precision_value, recall_value, beta=BETA_FOR_THRESHOLD)
        balanced_value = balanced_accuracy_score(y_true_array, predictions)
        mcc_value = matthews_corrcoef(y_true_array, predictions)
        metrics = {
            'threshold': float(threshold_value),
            'precision': float(precision_value),
            'recall': float(recall_value),
            'f1': float(f1_value),
            'f2': float(f2_value),
            'balanced_accuracy': float(balanced_value),
            'mcc': float(mcc_value),
            'recall_target_met': bool(recall_value >= TARGET_RECALL),
        }
        all_metrics.append(metrics)
        threshold_rows.append([threshold_value, precision_value, recall_value, f1_value, f2_value, balanced_value, mcc_value])

    # NEW: Recall-target constraint selection
    # Among thresholds meeting both criteria, pick the lowest one
    candidate_metrics = [
        m for m in all_metrics
        if m['recall'] >= TARGET_RECALL and m['precision'] >= MIN_PRECISION_FLOOR
    ]

    if candidate_metrics:
        # Pick lowest threshold that meets criteria (maximizes sensitivity)
        best_metrics = min(candidate_metrics, key=lambda m: m['threshold'])
    else:
        # Fallback: best recall that doesn't drop precision below half the floor
        fallback_candidates = [
            m for m in all_metrics
            if m['precision'] >= MIN_PRECISION_FLOOR * 0.5
        ]
        best_metrics = max(fallback_candidates, key=lambda m: m['recall'])

    return float(best_metrics['threshold']), best_metrics, np.asarray(threshold_rows, dtype=np.float32)
```

- [ ] **Step 2: Commit**

```bash
git add colab/notebook.ipynb
git commit -m "$(cat <<'EOF'
feat: update threshold selection to use recall-target constraint with precision floor

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Verify the Notebook Runs End-to-End

**Files:**
- Run: `colab/notebook.ipynb` (all cells)

- [ ] **Step 1: Verify feature dimension prints correctly**

After running cell 6, confirm the output shows `Feature dimension: 25` (was previously 14). If it's different, check that `len(FEATURE_NAMES)` matches.

- [ ] **Step 2: Verify training loop uses TverskyFocalLoss**

After running cells 8 through the training loop, confirm the loss output shows Tversky-style behavior (asymmetric gradient updates — FN should produce larger gradients than FP for similar confidence levels). Check that `TARGET_RECALL = 0.80` is used in threshold selection by printing it before the training loop.

- [ ] **Step 3: Commit any fixes**

If there are bugs found during verification (e.g., shape mismatches, NaN losses), fix them and commit with a descriptive message.

---

## Verification Checklist

After all tasks complete:

1. **Feature dimension**: `FEATURE_DIM` should be 25 (was 14)
2. **Tversky parameters**: Alpha=0.7, Beta=0.3 logged or verifiable in code
3. **Recall target**: TARGET_RECALL = 0.80 visible in cell 11
4. **Precision floor**: MIN_PRECISION_FLOOR = 0.85 visible in cell 11
5. **Learning rate**: lr = 1e-4 visible in adamw_kwargs
6. **Test metrics output**: Recall >= 0.80, Precision >= 0.85, F1 >= 0.74
