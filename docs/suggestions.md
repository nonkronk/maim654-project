# Deep Analysis & Improvement Suggestions
## Electricity Theft Detection — Trustworthy AI Project

---

## 1. Executive Diagnosis: Why the Results Are Poor

After a deep inspection of the notebook outputs, dataset, architecture, and training pipeline, the following critical root-cause issues were identified:

| Issue | Evidence | Impact |
|---|---|---|
| **Only 12 epochs of training** | Loss still falling at epoch 12 (1.178 → 1.064), no convergence | Model severely under-trained |
| **Batch size 16,384 is too large** | Gradient signal averaged over too many samples | Slow convergence, poor generalization |
| **Only 2 input features** (load + hour_sin) | 48 features claimed in the paper but only 2 in code | Under-powered representation |
| **Synthetic theft is trivially distinguishable** | Flat-line→near-zero; Drop→50–80% scale; Spike→2–3× mean | Model learns statistical shortcuts |
| **Single meter (MT_362 from UCI)** | Single consumer, narrow dynamic range, many zero-readings | Limited diversity; not representative |
| **Feature engineering too minimal** | Only sinusoidal encoding for hour — no daily stats, trend, volatility | Poor feature quality |
| **SHAP reveals correct temporal ordering** | All SHAP values are load features, none are hour_sin | hour_sin adds no useful signal |
| **Threshold fixed at 0.5** | Class-imbalanced dataset (15% theft) needs optimized threshold | Artificially low precision (0.31) |
| **Only 32 SHAP background samples** | Background set of 32, test set of 16, nsamples=20 | SHAP results statistically unreliable |
| **F1-Score = 0.405 for Theft class** | Precision 0.31 / Recall 0.59 — model guesses "theft" too liberally | False alarm rate extremely high |

---

## 2. Dataset Assessment

### Current Dataset: UCI Electricity Load Diagrams (Single Meter MT_362)

**Problems:**
- MT_362 is a commercial/industrial meter with extreme values (std = 9,681 W). The distribution is bimodal (many zero-readings during off-hours).
- Using a **single consumer's time series** means the model has no cross-consumer generalization.
- The synthetic theft labels are applied to MT_362's own normal windows — theft samples have the same base load profile as normal ones but with artificial transformations. This is too easy for any model to detect.

### Are We Using the Wrong Dataset?

The UCI Electricity Load Diagrams is not ideal for this task, but not fatally wrong. The issue is **how** we use it.

**Better approach with the same dataset:**
- Use **multiple meters** (the dataset has 371 columns). Sampling from diverse meters creates a richer normal distribution.
- Apply theft to some meters while keeping others clean, simulating realistic cross-consumer fraud detection.

**Ideal alternative dataset:** The **SGCC (State Grid Corporation of China) dataset** from Kaggle (`sgcc-data`) is a real-world electricity theft detection benchmark with labeled data (435 actual theft users + normal users). This is the gold standard for academic papers in this domain.

---

## 3. Improvement Roadmap (Prioritized)

### Priority 1: Fix Training Fundamentals (Quick Wins, High Impact)

#### 1.1 Increase Epochs and Enable Proper Early Stopping

```python
# BEFORE
epochs = 12
patience = 4

# AFTER
epochs = 100          # Let early stopping decide
patience = 10         # Give model time to improve
min_delta = 1e-4
```

The loss was still declining at epoch 12. The model never converged. Use early stopping based on **validation set F1**, not training loss.

#### 1.2 Add Validation Split

```python
# Split train into train/val (e.g., 70/10/20 instead of 80/20)
split_train = int(len(X_seq) * 0.70)
split_val   = int(len(X_seq) * 0.80)

X_train_raw = X_seq[:split_train]
X_val_raw   = X_seq[split_train:split_val]
X_test_raw  = X_seq[split_val:]
```

Without a validation set, early stopping on training loss overfits to training noise.

#### 1.3 Reduce Batch Size for Better Gradient Signal

```python
# BEFORE (GPU)
BATCH_SIZE = 16384  # Way too large for 112k samples

# AFTER
BATCH_SIZE = 512    # Still GPU-friendly, much better gradient signal
```

Large batch sizes are known to cause "sharp minima" and poor generalization. For 112k samples, batch size 512 (~219 steps per epoch) provides good gradient diversity.

#### 1.4 Optimize Detection Threshold

```python
# BEFORE
preds = (probs >= 0.5).long()

# AFTER — find optimal threshold on validation set
from sklearn.metrics import f1_score
thresholds = np.linspace(0.1, 0.9, 81)
val_f1s = [f1_score(y_val, (val_probs >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(val_f1s)]
# Typically 0.3-0.4 for 15% class imbalance
```

With 15% positive class, a threshold of 0.5 is biased. The optimal threshold is lower.

#### 1.5 Add Learning Rate Scheduler

```python
# Add cosine annealing or ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
# Call scheduler.step(val_loss) each epoch
```

---

### Priority 2: Improve Feature Engineering (Medium Effort, High Impact)

The current approach uses only 2 features: `load` and `hour_sin`. The model has almost no ability to detect anomalies relative to expected consumption patterns.

#### 2.1 Add Rolling Statistical Features per Window

For each 24-hour window, compute:

```python
def build_features(X_load, X_hour):
    features = []
    for i in range(len(X_load)):
        w = X_load[i]
        hour = X_hour[i]

        load_norm  = w                                          # (24,)
        hour_sin   = hour                                       # (24,)
        hour_cos   = np.cos(2 * np.pi * np.arange(24) / 24)   # (24,)
        win_mean   = np.full(24, w.mean())
        win_std    = np.full(24, w.std())
        win_max    = np.full(24, w.max())
        win_min    = np.full(24, w.min())
        flatness   = np.full(24, w.std() / (w.mean() + 1e-8))  # CV = key theft indicator
        load_ratio = w / (w.mean() + 1e-8)                     # within-window relative load

        f = np.stack([
            load_norm, hour_sin, hour_cos,
            win_mean, win_std, win_max, win_min,
            load_ratio
        ], axis=-1)   # shape (24, 8)
        features.append(f)

    return np.array(features)   # (N, 24, 8)
```

**Why these features help:**
- **`hour_cos`**: Completes the cyclical encoding (sin alone creates ambiguity at 0h/12h).
- **`win_std`**: Flat-line tampering -> std ~= 0. This single feature nearly perfectly detects flat-line theft.
- **`win_mean`**: Helps the model understand relative deviations.
- **`load_ratio`**: Normalizes within-window temporal patterns.
- **`flatness`** (CV = std/mean): Directly captures the flat-line signature.

#### 2.2 Add Day-of-Week Encoding

```python
# Theft patterns may differ by weekday vs. weekend
dow = (start_timestamp_of_window // 24) % 7
dow_sin = np.full(24, np.sin(2 * np.pi * dow / 7))
dow_cos = np.full(24, np.cos(2 * np.pi * dow / 7))
```

#### 2.3 Use More Meters for Baseline Distribution

```python
# Instead of 1 meter, use top 20 by variance
top_meters = column_stats_sorted[:20]
# Concatenate their windows to form a richer normal distribution
```

---

### Priority 3: Improve Synthetic Theft Injection (Medium Effort, High Impact)

The current theft injection is too artificial and the model learns statistical shortcuts rather than temporal patterns.

#### 3.1 More Realistic Theft Types

```python
def inject_theft(window, rng):
    """Implement 5 realistic theft patterns."""
    theft_type = rng.randint(0, 5)

    if theft_type == 0:
        # Meter bypass: uniform scale reduction
        scale = rng.uniform(0.2, 0.7)
        return window * scale

    elif theft_type == 1:
        # Illegal connection: additive spike
        noise_amp = rng.uniform(1.5, 3.0) * window.mean()
        return window + noise_amp * rng.random(len(window))

    elif theft_type == 2:
        # Flat-line (complete meter bypass)
        constant = rng.uniform(0, 0.05 * window.max())
        return np.full_like(window, constant)

    elif theft_type == 3:
        # Partial reporting: report only a subset of hours
        mask = rng.choice([0.0, 1.0], size=len(window), p=[0.5, 0.5])
        return window * mask

    else:
        # Phase shift: move night consumption to day (sophisticated fraud)
        shift = rng.randint(3, 8)
        return np.roll(window, shift)
```

#### 3.2 Vary the Anomaly Rate

```python
# Try 10% anomaly rate (more realistic) with stratified splits
n_anomalies = 0.10   # 10% theft windows
```

#### 3.3 Add Noise to Normal Windows

```python
# Real meters have measurement noise
noise_level = 0.02 * window.max()
window_noisy = window + rng.normal(0, noise_level, size=window.shape)
```

---

### Priority 4: Improve Model Architecture (Higher Effort, Moderate Gain)

#### 4.1 Increase Capacity

```python
# BEFORE
TheftDetectionLSTM(input_size=2, hidden_size=64, num_layers=2)

# AFTER
TheftDetectionLSTM(input_size=8, hidden_size=128, num_layers=2, dropout=0.3)
```

#### 4.2 Add Bidirectional LSTM

```python
self.lstm = nn.LSTM(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=True,    # ADD THIS
    dropout=dropout if num_layers > 1 else 0.0
)
# classifier input_size = hidden_size * 2
```

Bidirectional LSTMs capture both forward and backward temporal dependencies, particularly useful for detecting anomalies relative to surrounding context.

#### 4.3 Consider 1D-CNN + LSTM Hybrid (State-of-the-Art)

```python
class CNNLSTMDetector(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_filters=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.lstm = nn.LSTM(num_filters, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)             # (B, F, T) for Conv1d
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)             # back to (B, T, C)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1]).squeeze(-1)
```

The CNN layers extract local consumption shape patterns before the LSTM models temporal evolution. This is the state-of-the-art approach in published electricity theft detection papers.

#### 4.4 Add Focal Loss for Class Imbalance

Standard `BCEWithLogitsLoss` with `pos_weight` can over-correct. Focal loss focuses training on hard-to-classify examples:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()
```

---

### Priority 5: Fix SHAP & Adversarial Components

#### 5.1 Fix SHAP for Academic Credibility

The current SHAP setup uses only 32 background samples and 16 test samples. For a publication-quality result:

```python
# Use minimum 200 background samples, 200 test samples
background_size = min(200, len(X_train_t))
test_size = min(200, len(X_test_t))
shap_nsamples = 100  # was 20

# Stratify: include both normal and theft in test_samples
theft_idx  = np.where(y_test == 1)[0][:100]
normal_idx = np.where(y_test == 0)[0][:100]
shap_idx   = np.concatenate([theft_idx, normal_idx])
test_samples = X_test_t[shap_idx]
```

Also generate **separate SHAP force plots** for:
1. A True Positive (correctly detected theft)
2. A True Negative (correctly classified normal)
3. A False Positive (incorrectly flagged as theft)

This provides genuine insight and satisfies the XAI academic requirement.

#### 5.2 Add Adversarial Defense (Adversarial Training)

The notebook tests FGSM attacks but does not implement any defense. Add **adversarial training**:

```python
def fgsm_attack(model, X, y, criterion, epsilon):
    X_adv = X.clone().detach().requires_grad_(True)
    loss = criterion(model(X_adv), y.float())
    loss.backward()
    X_adv = X_adv + epsilon * X_adv.grad.sign()
    return X_adv.detach()

# During training: 50% clean, 50% adversarial
for batch_X, batch_y in train_loader:
    optimizer.zero_grad()
    if rng.random() > 0.5:
        logits = model(batch_X)
    else:
        X_adv = fgsm_attack(model, batch_X, batch_y, criterion, epsilon=0.01)
        logits = model(X_adv)
    loss = criterion(logits, batch_y.float())
    loss.backward()
    optimizer.step()
```

---

## 4. Alternative Dataset Recommendation

If the goal is maximizing academic quality and realistic results, switch to the **SGCC dataset**:

```python
import kagglehub
path = kagglehub.dataset_download('sgcc-data-electricity-theft')
```

**Why SGCC is better:**
- Contains **real labeled theft instances** (435 theft users vs. 42,372 normal users)
- Residential electricity consumers in China
- Each user has 1,035 daily consumption readings
- Widely used in literature; results are directly comparable

**Usage pattern:**
```python
# Each row = one user, each column = one day of consumption
# Reshape into daily windows of 24-hour readings
# Labels: 0 = normal, 1 = theft (ground truth, not synthetic)
```

---

## 5. Expected Results After Improvements

| Metric | Current | After Priority 1+2 | After All Improvements |
|---|---|---|---|
| **F1-Score (Theft)** | 0.405 | ~0.60-0.70 | ~0.75-0.85 |
| **Precision** | 0.308 | ~0.55-0.65 | ~0.70-0.80 |
| **Recall** | 0.591 | ~0.65-0.75 | ~0.75-0.85 |
| **PR-AUC** | N/A (not computed) | ~0.65-0.75 | ~0.80-0.90 |
| **Training epochs** | 12 (stopped early) | 30-60 (converged) | 50-100 (converged) |

---

## 6. Minimum Changes for Passing Quality

If time is limited, implement **these 5 changes** for the biggest immediate gain:

1. **Reduce batch size to 512** — better gradient signal, faster convergence
2. **Increase epochs to 50** with validation-based early stopping
3. **Add hour_cos feature + win_std + load_ratio** — change `input_size=2` to `input_size=5`
4. **Optimize detection threshold** on validation set instead of using 0.5
5. **Fix SHAP budget** -> 200 background samples, 200 test samples, nsamples=100

**Estimated F1 improvement from these 5 changes alone: +0.15 to +0.25**

---

## 7. Code Changes Summary

### File: `colab/notebook.ipynb`

| Cell | Change | Lines affected |
|---|---|---|
| Cell 3 (Dataset) | Use top 20 meters by variance | `column_stats_sorted[:20]` |
| Cell 5 (Features) | Return 6 features from `create_time_windows()` | New feature stack |
| Cell 6 (Theft) | Replace 3 theft types with 5 (add partial reporting, phase shift) | `generate_synthetic_labels()` |
| Cell 7 (Split) | Add validation split (70/10/20) | `split_train`, `split_val` |
| Cell 8 (Model) | `input_size=6, hidden_size=128` | Constructor args |
| Cell 9 (Train) | `epochs=50, patience=10, BATCH_SIZE=512, scheduler` | Training loop |
| Cell 10 (Eval) | Find optimal threshold on val set | Threshold search loop |
| Cell 12 (SHAP) | `background_size=200, test_size=200, nsamples=100` | SHAP cell params |

---

## 8. Report / Paper Changes Required

The LaTeX report (`docs/report.tex`) currently has `[fill]` in all result tables. After implementing improvements:

1. **Table I (Classification Performance):** Fill in real results. Target F1 > 0.70.
2. **Table II (Adversarial Robustness):** Add adversarial training as a defense and show metric recovery.
3. **Section III-B (Synthetic Labels):** Describe all 5 theft types with proper mathematical notation.
4. **Section III-C (Architecture):** Update to match CNN-LSTM if upgraded.
5. **Section IV-C (Explainability):** Include force plots, not just the summary bar chart.

---

## 9. Priority Execution Order

```
Week 1 (Before Presentation):
  [1] Fix batch size, epochs, validation split
  [2] Fix detection threshold
  [3] Add hour_cos + win_std features
  [4] Fix SHAP budget
  [5] Run and collect metrics

Week 2 (Polish):
  [6] Add 5 theft types
  [7] Implement adversarial defense
  [8] Update report tables
  [9] Generate force plots
```

---

*Analysis conducted: 2026-04-21 | Notebook: colab/notebook.ipynb | Report: docs/report.tex*
