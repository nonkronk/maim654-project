# Course Project: Trustworthy & Advanced AI Systems

**Weightage:** 20% of Final Grade  
**Group Size:** 3 Students  
**Due Date:** Week 14 (Presentations and Final Submission)

---

## 1. Project Overview

The objective of this project is to bridge the gap between theoretical AI architectures and deployable, trustworthy systems. This group project (3 members per group) requires you to select one application domain from the course topics (AI in Energy Systems, AI in Finance, or AI in Autonomous Systems) and develop a complete AI pipeline that demonstrates practical integration of at least two advanced topics from the course: Explainable AI (XAI) / Interpretable Machine Learning, Adversarial Machine Learning, Generative AI (I & II), and Ethics of AI.

The project simulates a real research-to-deployment workflow: you will solve a concrete problem in your chosen domain, implement and evaluate the system in code, interpret and harden the model, and communicate everything clearly. It directly assesses CLO2 (practical application of advanced AI techniques) through the Colab implementation and CLO3 (research communication) through the report and presentation.

## 2. Domain Selection

Your project should target one of the applied domains discussed in the course.

- **Track A: Energy Systems** (e.g., predictive maintenance, load forecasting, anomaly detection).
- **Track B: Financial Technology** (e.g., credit-risk prediction, credit card fraud detection, algorithmic trading).
- **Track C: Autonomous Systems** (e.g., object/anomaly detection, sensor fusion algorithms, adversarial attacks on traffic sign detection).

## 3. Technical Requirements

Define a specific, measurable problem in the chosen domain. Review available recent works from papers, blogs, or tutorials that relate to your chosen topic. Train/evaluate a baseline supervised or unsupervised model. Then, integrate **two** of the following AI topics:

1. **Explainability / Interpretability**: Implement post-hoc explainers (SHAP/LIME) to justify specific predictions, or use intrinsic "Glass Box" models (GAMs/ALE plots) instead of black boxes.
2. **Adversarial Robustness**: Generate adversarial examples (e.g., FGSM, PGD), measure drop in performance, and implement one defense (e.g., adversarial training, input preprocessing).
3. **Algorithmic Fairness**: Audit the model for demographic parity or equalized odds using a protected attribute, and implement a mitigation strategy (e.g., threshold optimization).
4. **Generative AI / LLM Integration**: Use VAEs/GANs to generate synthetic data for an imbalanced class, or build a ReAct Agent loop to solve a multi-step logic problem within your domain.

### Technical Constraints

- Use Google Colab only (free tier is sufficient).
- Public datasets only (Kaggle, UCI, Hugging Face, etc.).
- Libraries: any open-source library.
- Model size should be runnable on Colab GPU.

---

## 4. Project Deliverables

### Deliverable 1: The Codebase (Jupyter Notebook)

- Clean, well-documented, and reproducible code.
- Include visualizations and results.
- The code must execute successfully from end-to-end.

### Deliverable 2: The Technical Report (IEEE Format)

- **Length:** 3 to 5 pages, adhering to the standard IEEE two-column conference format.
- **Structure:**
	- Abstract & Introduction: Problem statement and dataset description.
	- Methodology: Detailed mathematical and architectural description of your advanced constraints (e.g., "How we applied SHAP").
	- Experiments & Results: Metrics beyond just accuracy (e.g., Precision-Recall curves, Pinball Loss).
	- Discussion: The trade-offs encountered (e.g., Accuracy vs. Fairness).
	- References: Minimum of 3 academic sources mapping to the state-of-the-art.

### Deliverable 3: Final Presentation (Week 12)

- Duration: 30 minutes (20 minutes presentation + 10 minutes Q&A).
- Focus heavily on the challenges faced and why your model behaves the way it does, rather than just reading the code on screen.
- Every group member must speak and be prepared to defend the architectural decisions.

### Grading Breakdown (Total 20% of course grade)

- Colab Notebook & Code Quality: 30%
- Report: 30%
- Presentation: 40%

---

## Final Project Grading Rubric (100 Total Points)

### 1. Colab Notebook & Code Quality (30%)

**Focus:** Is the code functional, reproducible, and does it successfully implement the advanced AI constraints?

| Criteria | Excellent (Full Marks) | Satisfactory (Partial Marks) | Needs Improvement (Low Marks) |
| --- | --- | --- | --- |
| **Advanced Implementation** | Flawlessly integrates two advanced constraints (e.g., XAI, Fairness). Logic is mathematically sound and perfectly suited to the domain. | Implements the constraints, but with minor logical flaws or sub-optimal library choices. | Fails to implement the required advanced constraints, or the implementation is broken. |
| **Reproducibility & Structure** | Notebook runs end-to-end without errors. Code is modular, highly documented, and data pipelines are clear. | Notebook runs with minor manual intervention. Comments are present but lack deep technical explanation. | Code throws errors, relies on hidden local files, or lacks basic structural comments. |

### 2. Technical Report [IEEE Format] (30%)

**Focus:** Can the students articulate their methodology, synthesize academic literature, and analyze trade-offs?

| Criteria | Excellent (Full Marks) | Satisfactory (Partial Marks) | Needs Improvement (Low Marks) |
| --- | --- | --- | --- |
| **Methodology & Depth** | Exceptionally clear justification of the architecture. Complex math/algorithms are explained accurately and cited properly. | Methodology is explained, but relies too heavily on generic descriptions rather than deep technical insight. | Explanations are superficial, inaccurate, or fail to connect the architecture to the chosen domain. |
| **Critical Analysis** | Provides a profound discussion of trade-offs (e.g., Accuracy vs. Fairness). Identifies limitations critically. | Discusses results adequately but lacks a deep dive into why the model behaved the way it did. | Only reports basic metrics (like accuracy) without analyzing limitations or real-world implications. |

### 3. Presentation & Defence (40%)

**Focus:** Can the students communicate complex AI concepts to an audience and defend their engineering choices under scrutiny?

| Criteria | Excellent (Full Marks) | Satisfactory (Partial Marks) | Needs Improvement (Low Marks) |
| --- | --- | --- | --- |
| **Clarity & Communication** | Delivery is highly engaging, timed perfectly, and distills complex architectures into clear, understandable visuals. | Delivery is clear but may rely too much on reading slides. Visuals are adequate but not highly illustrative. | Presentation is disorganized, confusing, or fails to cover the core technical achievements. |
| **Q&A Defence & Teamwork** | The member contributed clearly to the project. Answers to technical questions are precise, confident, and demonstrate good underlying knowledge. | The member answers questions adequately, but may hesitate or give surface-level responses. | Unable to defend architectural choices. Answers reveal a lack of understanding of the underlying code. |


