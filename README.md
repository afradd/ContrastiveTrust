# ContrastiveTrust

> **Physics-Guided Contrastive Learning for Zero-Shot Industrial Control System Anomaly Detection**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![IEEE](https://img.shields.io/badge/Research-IEEE-orange.svg)]()

---

## Overview

ContrastiveTrust is a self-supervised representation learning framework for Industrial Control System (ICS) anomaly detection.

Unlike conventional supervised approaches, the framework is trained **only on normal operating data** and detects previously unseen cyber-attacks without requiring attack labels during training.

The proposed method combines:

- Physics-guided feature engineering
- Contrastive representation learning
- Time-series encoding
- Zero-shot anomaly detection
- Cross-dataset evaluation

The framework is designed for realistic deployment where attack data are scarce or unavailable.

---

# Features

- Self-supervised training
- Zero-shot anomaly detection
- Physics-aware preprocessing
- Dataset-agnostic pipeline
- Modular PyTorch implementation
- IEEE reproducible experiments
- Cross-domain evaluation
- Publication-quality visualization

---

# Repository Structure

```text
ContrastiveTrust/
│
├── src/
├── preprocessing/
├── tests/
├── notebooks/
├── paper/
├── artifacts/
├── data/
└── README.md
```

---

# Pipeline

```
Raw Sensor Data
      │
      ▼
Data Cleaning
      │
      ▼
Window Generation
      │
      ▼
Physics-guided Features
      │
      ▼
Contrastive Augmentation
      │
      ▼
Encoder Network
      │
      ▼
NT-Xent Loss
      │
      ▼
Embedding Space
      │
      ▼
Zero-shot Inference
      │
      ▼
Anomaly Score
```

---

# Installation

```bash
git clone https://github.com/afradd/ContrastiveTrust.git

cd ContrastiveTrust

pip install -r requirements.txt
```

---

# Dataset

This repository supports:

- HAI
- SWaT

Datasets are **not included** because of licensing restrictions.

Please place datasets inside

```
data/raw/
```

---

# Training

```bash
python train.py
```

---

# Evaluation

```bash
python test.py
```

---

# Results

The repository includes scripts for generating:

- Training loss
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Embedding Visualization
- Score Distribution

---

# Research Highlights

- Physics-guided learning
- Self-supervised representation learning
- Zero-shot anomaly detection
- Industrial Control Systems
- Cross-dataset generalization

---

# Citation

```bibtex
@article{contrastivetrust2026,
  title={ContrastiveTrust: Physics-Guided Contrastive Learning for Zero-Shot Industrial Control System Anomaly Detection},
  author={Afrad Ali Ahim},
  year={2026}
}
```

---

# License

MIT License

---

# Acknowledgements

- HAI Dataset
- SWaT Dataset
- PyTorch
- IEEE

---

# Author

**Afrad Ali Ahim**

Computer Science & Engineering

International Islamic University Chittagong