# Electricity Theft Detection with LSTM, XAI, and Adversarial Testing

**Date:** 2026-04-21
**Team:** 3 students (PLN-sponsored MS AI student at Georgia Tech)
**Track:** Energy Systems — Predictive Maintenance / Anomaly Detection

## Project Overview

Detect electricity theft using an LSTM time-series classifier trained on public load data with synthetic theft patterns. The project demonstrates two advanced AI topics: **Explainable AI (SHAP/LIME)** and **Adversarial Robustness (FGSM)**.

## Domain Context

PT PLN (Persero) loses billions annually to electricity theft — bypassed meters, illegal connections, tampered readings. This project simulates detection using public datasets as a proof-of-concept for real-world deployment at PLN substations and metering points.

## Dataset

- **Primary:** UCI Electricity Load Diagrams Dataset (hourly consumption data, 2014-2018)
- **Synthetic Labels:** Simulated theft patterns injected into normal load data:
  - Sudden drops (bypassed meter) — reduce consumption by 50-80% during specific hours
  - Unusual spikes (illegal connections) — add 2-3x normal baseline during peak hours
  - Flat-line periods (tampered reading) — set readings to zero or constant for extended periods

## Architecture

```
UCI Load Data → Preprocessing & Feature Engineering → LSTM Classification → SHAP Explanation → FGSM Adversarial Testing
```

### Data Pipeline
1. Download UCI load diagrams CSV data
2. Normalize and create sliding time windows (sequence of hourly values)
3. Engineer features: hour-of-day, day-of-week, rolling averages
4. Inject synthetic theft patterns as "anomaly" class labels
5. Train/test split by time (not random) to simulate real-world temporal split

### Model Architecture
- 2-layer LSTM (hidden size: 64)
- Dense(64, ReLU) → Dense(1, sigmoid) for binary classification
- ~50K parameters — easily runs on Colab GPU
- Class-weighted loss function for imbalanced data

### XAI Integration
- SHAP DeepExplainer for feature importance analysis
- Force plots for specific predictions to justify model decisions
- LIME for additional interpretability on individual samples

### Adversarial Testing
- FGSM (Fast Gradient Sign Method) perturbation with epsilon values: 0.01, 0.05, 0.1
- Measure F1-score drop and false negative rate increase under attack
- Compare robustness across different threat levels — report exact % metrics at each epsilon

## Evaluation Metrics
- Precision, Recall, F1-score (not just accuracy)
- Precision-Recall curve
- False negative rate (missed thefts — critical metric)
- Adversarial robustness score (% performance drop under FGSM attack)

## Deliverables

### 1. Colab Notebook
End-to-end executable notebook covering: data loading, preprocessing, LSTM training, SHAP analysis, adversarial testing, visualization and results.

### 2. IEEE Report (3-5 pages)
Structure:
- Abstract & Introduction — problem statement, dataset description
- Methodology — LSTM architecture math, SHAP application, FGSM methodology
- Experiments & Results — PR curves, F1 scores, adversarial robustness results
- Discussion — accuracy vs robustness trade-offs, synthetic label limitations
- References — minimum 3 academic sources

### 3. Presentation (30 min)
- 20 min presentation + 10 min Q&A
- Focus on challenges faced and model behavior analysis
- All 3 team members speak and defend architectural decisions

## Constraints
- Google Colab only (Pro tier available)
- Public datasets only (Kaggle, UCI, HuggingFace)
- Two advanced topics: XAI (SHAP/LIME) + Adversarial Robustness (FGSM)
- Model runnable on Colab GPU within resource limits

## Technical Stack
- Python, PyTorch/LSTM, SHAP, CleverHans/AdversarialRobustnessToolkit
- UCI Load Diagrams Dataset
