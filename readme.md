# MAGI-Vul: Multi-Scale Gated Interaction for Robust Function-Level Vulnerability Detection

This repository contains the implementation of **MAGI-Vul**, a Mamba-based framework for function-level vulnerability detection.

MAGI-Vul is designed to address a key limitation of existing vulnerability detection approaches: their difficulty in effectively modeling heterogeneous vulnerability signals that appear at different semantic granularities within source code. To address this issue, MAGI-Vul introduces a multi-scale representation framework that combines dynamic sequence modeling, adaptive feature fusion, and cross-level interaction enhancement.

The work is presented in:

**MAGI-Vul: Multi-Scale Gated Interaction for Robust Function-Level Vulnerability Detection**

---

## Overview

Function-level vulnerability detection aims to determine whether a software function contains security vulnerabilities.

MAGI-Vul builds upon the Mamba architecture and incorporates three complementary components:

- **Multi-Scale Feature Pyramid (MFP)** for capturing vulnerability patterns at different semantic scales.
- **Adaptive Gated Fusion (AGF)** for dynamically integrating multi-scale representations.
- **Interaction Enhancement Module (IEM)** for strengthening information exchange between global and scale-specific features.

The overall architecture enables MAGI-Vul to learn richer vulnerability-aware representations while maintaining the efficiency advantages of selective state-space models.

Experimental results on the **CVEfixes** benchmark demonstrate that MAGI-Vul consistently outperforms existing deep learning baselines and achieves state-of-the-art performance in both overall detection and CWE-specific evaluation settings.

---

## Dataset Preparation

### 1. Download Dataset

All experiments are conducted on the **CVEfixes** dataset.

Please download the dataset from the official repository:

- CVEfixes

### 2. Data Processing

MAGI-Vul follows the function-level vulnerability detection setting.

Each sample contains:

- Function source code
- Vulnerability label
- CWE category information (for CWE-specific evaluation)

After preprocessing, split the dataset into training, validation, and testing sets following the experimental protocol described in the paper.

---

## Environment Setup

Create a Python environment and install the required dependencies.

```bash
pip install torch
pip install numpy
pip install pandas
pip install scikit-learn
pip install tqdm
```

Recommended environment:

```text
Python >= 3.10
PyTorch >= 2.0
CUDA >= 11.8
```

---

## Model Architecture

MAGI-Vul consists of four major components:

### 1. Mamba Backbone

The backbone employs stacked Mamba blocks to model long-range dependencies and contextual relationships in source code.

### 2. Multi-Scale Feature Pyramid (MFP)

The MFP module generates representations at multiple semantic scales, allowing the model to capture both local vulnerability patterns and higher-level contextual information.

### 3. Adaptive Gated Fusion (AGF)

The AGF module learns dynamic gating coefficients to selectively aggregate multi-scale features according to their importance for vulnerability prediction.

### 4. Interaction Enhancement Module (IEM)

The IEM module utilizes attention-based interactions to strengthen communication between backbone features and multi-scale representations, improving feature expressiveness and robustness.

---

## Running Experiments

### Train MAGI-Vul

```bash
python train.py
```

The training script automatically performs:

- Dataset loading
- Data preprocessing
- Model training
- Validation-based model selection
- Final testing

---

## Baseline Models

The repository also includes implementations of the baseline architectures used in the paper:

- Transformer
- iTransformer
- Mamba

These models are used for comparative evaluation under the same experimental settings.

---

## Evaluation Metrics

Following prior vulnerability detection studies, we report:

- Accuracy
- Precision
- Recall
- F1-score

In addition, CWE-specific evaluations are conducted to assess robustness across representative vulnerability categories.

---

## Main Results

### Overall Vulnerability Detection

MAGI-Vul achieves:

- **F1-score: 0.91**
- **Recall: 0.99**

and outperforms all compared baselines on the CVEfixes dataset.

### CWE-Specific Evaluation

MAGI-Vul demonstrates strong robustness across representative vulnerability categories and achieves:

- **Macro-F1: 0.84**

which is the best overall result among all evaluated methods.

---

## Ablation Study

To investigate the contribution of each component, we conduct ablation experiments by progressively removing:

- Multi-Scale Feature Pyramid (MFP)
- Adaptive Gated Fusion (AGF)
- Interaction Enhancement Module (IEM)

Results show that each component contributes positively to the final performance, and the complete MAGI-Vul architecture achieves the highest F1-score.

---

## Reproducibility

For reproducibility, all experiments use a fixed random seed and identical dataset splits across different models.

The training pipeline automatically:

- Saves the best-performing checkpoint.
- Records training statistics.
- Reports testing performance.
- Generates experimental logs.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{magivul2026,
  title={MAGI-Vul: Multi-Scale Gated Interaction for Robust Function-Level Vulnerability Detection},
  author={Anonymous},
  year={2026}
}
```

---

## License

This repository is released for academic research purposes only.
