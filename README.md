# Electricity Theft Detection with Trustworthy AI

A group project for the [Master in Artificial Intelligence (UNITEN)](https://www.uniten.edu.my/programme/postgraduate/master-in-artificial-intelligence-by-coursework-project), developed for the MAIM654: Advanced Artificial Intelligence course taught by Dr Ahmed Mubarak.

## Project Context

This repository fulfills the requirements in [project-statement.md](project-statement.md) for the project titled **"Electricity Theft Detection with Trustworthy AI"**.

- **Track:** Energy Systems
- **Theme:** Electricity theft detection from smart meter load profiles

## Problem Statement

Electricity theft causes substantial utility losses and undermines grid reliability. This project develops a trustworthy AI pipeline that detects suspicious consumption behavior from time-series load data, with a focus on practical deployment properties:

- Temporal train/test splits
- Interpretable predictions
- Robustness under adversarial perturbations

## Advanced AI Topics

This project integrates at least two advanced topics from the course:

**1. Explainable AI (XAI)**
- SHAP for global feature attributions
- LIME for local, per-sample explanations

**2. Adversarial Machine Learning**
- FGSM attack generation across multiple epsilon levels
- Performance degradation analysis and robustness reporting

**Optional extensions:**
- Fairness auditing and mitigation
- Synthetic data generation (GAN/VAE) for class imbalance

## Technical Pipeline

```mermaid
flowchart LR
    A[UCI Electricity Load Data] --> B[Cleaning + Feature Engineering]
    B --> C[Sliding Time Windows]
    C --> D[Synthetic Theft Pattern Injection]
    D --> E[LSTM Binary Classifier]
    E --> F[Evaluation: Precision / Recall / F1 / PR-AUC]
    E --> G[XAI: SHAP and LIME]
    E --> H[Adversarial Test: FGSM]
    H --> I[Robustness Report]
```

## Dataset Strategy

**Primary dataset:** [UCI Electricity Load Diagrams 2011–2014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)

**Labeling approach** — synthetic theft scenarios injected into normal sequences:

- **Sudden drops** — meter bypass behavior
- **Unusual spikes** — illegal tapping behavior
- **Flat-line patterns** — meter tampering behavior

## Repository Layout

```
├── README.md
├── project-statement.md
├── colab/
│   └── notebook.ipynb
└── docs/
    ├── report/
    │   └── ieee-report-skeleton.md
    └── superpowers/
        ├── plans/2026-04-21-electricity-theft-detection.md
        └── specs/2026-04-21-electricity-theft-detection.md
```

## Deliverables

**1. Colab Notebook** — [`colab/notebook.ipynb`](colab/notebook.ipynb)
End-to-end reproducible notebook covering data ingestion, model training, explainability, adversarial testing, and visual outputs.

**2. IEEE-Style Technical Report (3–5 pages)** — [`docs/report/ieee-report-skeleton.md`](docs/report/ieee-report-skeleton.md)
Covers methodology and equations, experimental results, robustness findings, and trade-off discussion (accuracy vs. robustness, interpretability vs. complexity).

**3. Final Presentation**
20-minute presentation + 10-minute Q&A, with emphasis on engineering decisions and model behavior analysis.

## Progress Roadmap

- [x] Project scope selection (Energy Systems)
- [x] Initial specification and implementation plan drafted
- [ ] Build complete Colab notebook skeleton
- [ ] Data loading and exploratory analysis
- [ ] Synthetic theft label generation
- [ ] LSTM baseline training and evaluation
- [ ] SHAP and LIME integration
- [ ] FGSM robustness testing
- [ ] Results consolidation and report writing
- [ ] Slide deck and defense preparation

## Evaluation Metrics

**Primary metrics:** Precision · Recall · F1-score · PR-AUC

**Trustworthiness metrics:**
- False negative rate on theft cases
- F1 drop under FGSM across epsilon levels
- Explanation consistency on correctly classified anomalies

## Planned Visualizations

- Baseline load profile plots
- Class distribution charts
- Confusion matrix
- Precision–recall curve
- SHAP summary and force plots
- Robustness curve: metric vs. epsilon

## Reproducibility

- **Platform:** Google Colab (free/standard GPU tier)
- **Data:** Public datasets only
- **Goal:** Clean, modular, end-to-end executable code

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch shap lime
```

## References

1. Asghar, M. R., et al. "Smart meter data privacy and security: A survey." *IEEE Communications Surveys & Tutorials*, 2017. [doi:10.1109/COMST.2017.2720195](https://doi.org/10.1109/COMST.2017.2720195)
2. Jokar, P., Arianpoo, N., and Leung, V. C. M. "Electricity theft detection in AMI using customers' consumption patterns." *IEEE Transactions on Smart Grid*, 2016. [doi:10.1109/TSG.2015.2425222](https://doi.org/10.1109/TSG.2015.2425222)
3. Lundberg, S. M., and Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." *NeurIPS*, 2017. [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
4. Goodfellow, I. J., Shlens, J., and Szegedy, C. "Explaining and Harnessing Adversarial Examples." *ICLR*, 2015. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)

**Library documentation:** [SHAP](https://shap.readthedocs.io) · [LIME](https://lime-ml.readthedocs.io) · [PyTorch](https://pytorch.org/docs/stable/index.html) · [Google Colab](https://colab.research.google.com)

## License

This repository is for academic coursework and research prototyping. Dataset and library licenses should be respected per their original terms.
