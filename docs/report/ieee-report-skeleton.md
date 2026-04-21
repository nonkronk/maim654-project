# IEEE Report Draft - Electricity Theft Detection with Trustworthy AI

Paper type: Course Project Report (3 to 5 pages, IEEE style)  
Course: Advanced Artificial Intelligence  
Institution: Universiti Tenaga Nasional (UNITEN)  
Program: Master in Artificial Intelligence (Coursework and Project)  
Supervisor: Dr Ahmed Mubarak  
Date: April 2026

Project repository context:
- README: [../../README.md](../../README.md)
- Project statement: [../../project-statement.md](../../project-statement.md)
- Implementation notebook: [../../colab/notebook.ipynb](../../colab/notebook.ipynb)

## Title

Electricity Theft Detection from Smart Meter Time-Series Data using LSTM with Explainable AI and Adversarial Robustness Testing

## Authors

- Student 1 Name, UNITEN, email
- Student 2 Name, UNITEN, email
- Student 3 Name, UNITEN, email

## Abstract

Electricity theft remains a major operational and economic challenge for utility providers. This report presents a trustworthy AI pipeline for theft detection from smart meter time-series data using a long short-term memory (LSTM) classifier. The approach uses public electricity load data and synthetic theft pattern injection to construct a practical binary classification problem under constrained academic resources. To align with deployment trust requirements, the model is evaluated beyond standard predictive performance through two advanced components: explainable AI (SHAP and LIME) and adversarial robustness testing (FGSM). The explainability stage identifies which temporal patterns influence theft predictions, while the robustness stage quantifies metric degradation under bounded perturbations. The full workflow is designed for end-to-end execution in Google Colab and emphasizes reproducibility, interpretability, and stress testing. This structure supports energy-domain decision making where missed anomalies and opaque predictions carry significant practical risk.

Keywords: electricity theft detection, LSTM, explainable AI, SHAP, LIME, adversarial machine learning, FGSM, trustworthy AI

## I. Introduction

Electricity theft includes meter tampering, illegal connections, and bypass behavior that can reduce utility revenue and degrade grid planning accuracy. In practical environments, detection systems must not only be accurate but also explainable and robust to input manipulations. This report focuses on a trustworthy AI framing for theft detection in which performance, interpretability, and robustness are evaluated together.

The task is defined as follows: given hourly smart meter load sequences, classify each sequence as normal consumption or suspected theft. The project follows a deployment-oriented workflow using temporal train/test separation, avoiding random splitting that can overestimate real-world performance.

The main contributions are threefold. First, we build an end-to-end LSTM-based detection pipeline over public data with synthetic theft pattern generation. Second, we apply SHAP and LIME to provide global and local explanation layers for model outputs. Third, we quantify adversarial sensitivity under FGSM perturbations and analyze the resulting trust-performance trade-offs.

## II. Related Work

Prior electricity theft detection studies span rule-based, statistical, and machine learning methods. Classical techniques, such as support vector machines and tree-based ensembles, are often effective under static feature sets but may struggle to represent temporal dependencies in load profiles. Deep sequence models, including recurrent neural networks and LSTM variants, are more suitable for capturing periodicity and abrupt usage shifts in smart meter data.

In parallel, explainable AI has become increasingly important in energy-domain analytics where operational teams require transparent rationale before taking action against suspected theft cases. Post-hoc explanation tools such as SHAP and LIME are widely used to audit model behavior and improve stakeholder confidence.

Adversarial machine learning literature shows that neural models can be sensitive to small, structured perturbations. Although much of this work is image-centric, robustness testing remains relevant for time-series models because malicious or noisy perturbations in meter signals can alter classification outcomes.

## III. Methodology

### A. Dataset and Preprocessing

The project uses the UCI Electricity Load Diagrams dataset. Raw meter series are cleaned for missing values, normalized using training-set statistics only, and transformed into sliding windows to form model inputs. A chronological split is applied to better simulate deployment conditions.

### B. Synthetic Theft Label Generation

Because real theft labels are limited in many public datasets, synthetic anomaly injection is used to generate training targets. Three behavior types are simulated: sudden load drops, unusual spikes, and flat-line tampering. Let $x \in \mathbb{R}^{T}$ denote a normal sequence and $\tilde{x}$ denote the transformed sequence after injection. Binary labels are defined as $y=1$ for injected theft-like patterns and $y=0$ otherwise.

### C. LSTM Classifier

The classifier is a two-layer LSTM followed by a dense sigmoid output layer. For input sequence $x=[x_1,\dots,x_T]$, hidden states are updated as

$$
h_t, c_t = \mathrm{LSTM}(x_t, h_{t-1}, c_{t-1}),
$$

and theft probability is computed by

$$
\hat{y} = \sigma(Wh_T + b).
$$

Class-weighted binary cross-entropy is used to address imbalance:

$$
\mathcal{L} = -\frac{1}{N}\sum_i\left[w_1 y_i\log(\hat{y}_i) + w_0(1-y_i)\log(1-\hat{y}_i)\right].
$$

### D. Explainability Layer

SHAP is used for global feature attribution and per-sample contribution analysis across time-lagged inputs. LIME provides localized, interpretable surrogate explanations for selected borderline or high-impact predictions.

### E. Adversarial Robustness Layer

FGSM perturbation is applied to the input features using

$$
x_{adv} = x + \epsilon\,\mathrm{sign}(\nabla_x\mathcal{L}(\theta, x, y)).
$$

Performance is evaluated across $\epsilon \in \{0.01, 0.05, 0.10\}$ and compared with clean-input results.

## IV. Experiments and Setup

Experiments run in Google Colab using Python, PyTorch, scikit-learn, SHAP, and LIME. The baseline configuration used for initial runs is: window size 24, batch size 64, learning rate 0.001, 30 epochs, hidden size 64, two LSTM layers, and an 80/20 chronological split.

Evaluation focuses on precision, recall, F1-score, PR-AUC, false negative rate, and robustness drop under attack. Robustness drop is reported as

$$
\Delta_{robust} = \frac{F1_{clean} - F1_{adv}}{F1_{clean}}.
$$

## V. Results and Discussion

### A. Baseline Classification Performance

| Metric | Value |
| --- | --- |
| Precision | [fill] |
| Recall | [fill] |
| F1-score | [fill] |
| PR-AUC | [fill] |
| False Negative Rate | [fill] |

### B. Explainability Findings

Preliminary interpretation should report whether the most influential temporal features match expected theft signatures (sudden drops, abnormal spikes, and flat segments). Include one global SHAP summary and at least one local explanation for both SHAP and LIME.

Figure placeholders:
- Fig. 1: SHAP global feature importance summary
- Fig. 2: SHAP force or waterfall plot for one true-positive theft sample
- Fig. 3: LIME local explanation for one borderline sample

### C. Adversarial Robustness Findings

| Epsilon | Precision | Recall | F1-score | Relative F1 Drop |
| --- | --- | --- | --- | --- |
| 0.00 (clean) | [fill] | [fill] | [fill] | 0.00 |
| 0.01 | [fill] | [fill] | [fill] | [fill] |
| 0.05 | [fill] | [fill] | [fill] | [fill] |
| 0.10 | [fill] | [fill] | [fill] | [fill] |

### D. Trade-off Analysis

This section should explain how robustness testing changes confidence in deployment readiness. In particular, discuss: (i) the balance between recall and false alarms, (ii) degradation trends under increasing $\epsilon$, and (iii) computational overhead introduced by explainability and robustness analysis.

### E. Threats to Validity

Synthetic labels may not represent all real-world theft strategies, and domain shift from public datasets to utility-specific distributions can affect generalization. Additionally, FGSM represents a first-order attack model and may not cover all practical attack channels.

## VI. Ethical and Practical Considerations

False positives can create customer-impact risk; therefore, predictions should support, not replace, human investigation workflows. Data handling must follow privacy principles for meter traces. Future iterations should include fairness checks across demographic or regional usage segments to reduce unintended bias in enforcement-related decisions.

## VII. Conclusion and Future Work

This report presents a deployment-oriented trustworthy AI approach for electricity theft detection using LSTM classification, explainability analysis, and adversarial stress testing. The pipeline is reproducible in Colab and structured for academic evaluation and practical discussion. Future work includes testing on real labeled utility data, incorporating adversarial training defenses, and benchmarking against alternative sequence architectures such as temporal convolutional networks and transformers.

## Acknowledgment

This report is prepared as part of the Advanced Artificial Intelligence course project at UNITEN.

## References

[1] P. Jokar, N. Arianpoo, and V. C. M. Leung, "Electricity theft detection in AMI using customers' consumption patterns," IEEE Transactions on Smart Grid, vol. 7, no. 1, pp. 216-226, 2016. DOI: https://doi.org/10.1109/TSG.2015.2425222

[2] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems (NeurIPS), 2017. URL: https://arxiv.org/abs/1705.07874

[3] I. J. Goodfellow, J. Shlens, and C. Szegedy, "Explaining and harnessing adversarial examples," in International Conference on Learning Representations (ICLR), 2015. URL: https://arxiv.org/abs/1412.6572

[4] M. R. Asghar, G. Dan, D. Miorandi, and I. Chlamtac, "Smart meter data privacy and security: A survey," IEEE Communications Surveys and Tutorials, vol. 19, no. 4, pp. 2820-2835, 2017. DOI: https://doi.org/10.1109/COMST.2017.2720195

[5] Add at least one 2021+ electricity theft or smart-grid anomaly detection paper from IEEE, Elsevier, or Springer.

## Finalization Checklist

- [ ] Replace all [fill] entries with notebook outputs
- [ ] Insert figure files and in-text citations (Fig. 1, Fig. 2, Fig. 3)
- [ ] Validate that claims in discussion are backed by reported metrics
- [ ] Convert to IEEE two-column final template prior to submission
- [ ] Add full author names and affiliations
