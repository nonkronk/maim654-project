# IEEE Report Draft - Electricity Theft Detection with Trustworthy AI

Paper type: Course Project Report (3 to 5 pages, IEEE style)
Course: Advanced Artificial Intelligence
Institution: Universiti Tenaga Nasional (UNITEN)
Program: Master in Artificial Intelligence (Coursework and Project)
Supervisor: Dr Ahmed Mubarak
Date: April 2026

## Title

Electricity Theft Detection from Smart Meter Time-Series Data using LSTM with Explainable AI and Adversarial Robustness Testing

## Authors

- Student 1 Name, UNITEN, email
- Student 2 Name, UNITEN, email
- Student 3 Name, UNITEN, email

## Abstract

Electricity theft is a persistent challenge for utility providers, causing major financial losses and operational instability. This project presents a trustworthy machine learning pipeline for electricity theft detection using smart meter load profiles. We train a long short-term memory (LSTM) classifier on public electricity consumption data with synthetic theft scenarios to simulate realistic anomaly patterns. To improve trust and deployment readiness, we integrate two advanced AI components: explainable AI (SHAP and LIME) and adversarial robustness testing (FGSM). We evaluate performance using precision, recall, F1-score, and precision-recall curves, then measure degradation under perturbation strengths. Results demonstrate that the proposed model can detect suspicious patterns while exposing key decision factors and robustness limits. The framework is designed for end-to-end execution in Google Colab and aligns with practical constraints for educational and early deployment studies.

Keywords: electricity theft detection, LSTM, explainable AI, SHAP, LIME, adversarial machine learning, FGSM, trustworthy AI

## I. Introduction

### A. Background and Motivation

Electricity theft includes meter tampering, illegal tapping, and bypassed metering behavior. Such activities reduce utility revenue, distort demand forecasting, and impact grid reliability.

### B. Problem Statement

Given hourly smart meter load sequences, classify each sample as normal usage or suspected theft behavior.

### C. Contributions

- Build an end-to-end electricity theft detection pipeline using public data.
- Train an LSTM classifier for time-series anomaly classification.
- Integrate explainability using SHAP and LIME to justify predictions.
- Evaluate adversarial robustness using FGSM and analyze metric degradation.
- Provide practical insights for trustworthy AI deployment in energy systems.

### D. Paper Organization

Section II reviews related work. Section III details methodology. Section IV presents experiments. Section V discusses results and trade-offs. Section VI concludes.

## II. Related Work

### A. Electricity Theft Detection

Summarize prior methods:
- Statistical and rule-based anomaly detection
- Classical ML methods (SVM, Random Forest, XGBoost)
- Deep learning approaches for smart meter sequences

### B. Explainable AI in Energy Systems

Discuss why explanation matters in high-impact utility operations and auditing.

### C. Adversarial Robustness for Time-Series Models

Describe how adversarial perturbations can alter predictions and why robustness testing is required before deployment.

## III. Methodology

### A. Dataset and Preprocessing

Dataset: UCI Electricity Load Diagrams Dataset.

Preprocessing steps:
- Missing value handling
- Normalization using training statistics only
- Sliding time window construction
- Time-aware split (train and test by chronology)

### B. Synthetic Theft Label Generation

We inject synthetic theft behavior to create binary labels:
- Sudden drops: multiply values by 0.2 to 0.5 for selected windows
- Unusual spikes: increase baseline by 2x to 3x at targeted intervals
- Flat-line tampering: replace segments with near-constant values

Let x in R^T be a normal sequence and x_tilde be transformed sequence after theft injection. Label y = 1 for injected anomalies and y = 0 for normal behavior.

### C. LSTM Classifier

Input sequence x = [x_1, x_2, ..., x_T] is passed to a two-layer LSTM:

h_t, c_t = LSTM(x_t, h_(t-1), c_(t-1))

y_hat = sigma(W h_T + b)

where y_hat in [0, 1] is theft probability.

Loss:

L = - (1/N) * sum_i [w_1 y_i log(y_hat_i) + w_0 (1 - y_i) log(1 - y_hat_i)]

Use class weighting (w_1, w_0) for imbalance.

### D. Explainability Module

SHAP:
- Compute feature attributions over time-lagged inputs.
- Report global importance and local explanation plots.

LIME:
- Generate local surrogate explanation for selected borderline predictions.

### E. Adversarial Robustness Module

FGSM perturbation:

x_adv = x + epsilon * sign(gradient_x L(theta, x, y))

Evaluate at epsilon in {0.01, 0.05, 0.10}. Compare performance against clean baseline.

## IV. Experiments and Setup

### A. Environment

- Platform: Google Colab
- Frameworks: Python, PyTorch, scikit-learn, SHAP, LIME
- Hardware: Colab CPU/GPU runtime

### B. Data Split and Hyperparameters

Template values to fill after final runs:
- Window size: [fill]
- Batch size: [fill]
- Learning rate: [fill]
- Epochs: [fill]
- LSTM hidden size: [fill]
- Number of LSTM layers: [fill]
- Train/test split ratio: [fill]

### C. Evaluation Metrics

- Precision
- Recall
- F1-score
- PR-AUC
- False negative rate (critical for missed theft)
- Robustness drop = (F1_clean - F1_adv) / F1_clean

## V. Results and Discussion

### A. Baseline Classification Performance

Table 1 template:

| Metric | Value |
| --- | --- |
| Precision | [fill] |
| Recall | [fill] |
| F1-score | [fill] |
| PR-AUC | [fill] |
| False Negative Rate | [fill] |

### B. Explainability Findings

Points to report:
- Most influential time steps/features from SHAP
- Example local explanation from LIME and operational interpretation
- Whether explanations align with expected theft behavior

### C. Adversarial Robustness Findings

Table 2 template:

| Epsilon | Precision | Recall | F1-score | Relative F1 Drop |
| --- | --- | --- | --- | --- |
| 0.00 (clean) | [fill] | [fill] | [fill] | 0.00 |
| 0.01 | [fill] | [fill] | [fill] | [fill] |
| 0.05 | [fill] | [fill] | [fill] | [fill] |
| 0.10 | [fill] | [fill] | [fill] | [fill] |

### D. Trade-off Analysis

Discuss:
- Accuracy vs robustness
- Sensitivity vs false alarms
- Interpretability depth vs computational overhead

### E. Threats to Validity

- Synthetic labels may not capture all real theft patterns
- Distribution gap between public dataset and utility-specific meter behavior
- Adversarial setting may not reflect full physical attack surface

## VI. Ethical and Practical Considerations

- Risk of false positives affecting legitimate customers
- Need for human-in-the-loop review before punitive action
- Data privacy and secure handling of consumption traces
- Bias checks for regional or socioeconomic usage patterns (future work)

## VII. Conclusion and Future Work

This project demonstrates a practical trustworthy AI pipeline for electricity theft detection using LSTM, explainability, and adversarial robustness testing. Future work includes validation on real labeled utility data, adversarial training for defense, and expansion to transformer-based sequence models.

## Acknowledgment

This report is prepared as part of the Advanced Artificial Intelligence course project at UNITEN.

## References (Initial Draft)

[1] P. Jokar, N. Arianpoo, and V. C. M. Leung, "Electricity theft detection in AMI using customers' consumption patterns," IEEE Transactions on Smart Grid, vol. 7, no. 1, pp. 216-226, 2016.

[2] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems (NeurIPS), 2017.

[3] I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in International Conference on Learning Representations (ICLR), 2015.

[4] M. R. Asghar, G. Dan, D. Miorandi, and I. Chlamtac, "Smart meter data privacy and security: A survey," IEEE Communications Surveys and Tutorials, vol. 19, no. 4, pp. 2820-2835, 2017.

[5] Add at least one recent (2021+) electricity theft or smart-grid anomaly detection paper.

## Writing Checklist

- [ ] Replace all [fill] placeholders with experiment outputs
- [ ] Add figure references and captions from notebook plots
- [ ] Convert to IEEE two-column template in final submission format
- [ ] Ensure all claims in discussion are supported by metrics
- [ ] Add minimum 3 strong academic references (already seeded with 4)
- [ ] Proofread for tense consistency and technical clarity
