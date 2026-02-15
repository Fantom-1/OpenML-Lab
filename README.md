# 🏛️ OpenMLLab: Machine Learning from First Principles

**Bridging the gap between Research Papers and Production-Ready Code.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![Build: Research-Grade](https://img.shields.io/badge/Build-Research--Grade-orange.svg)]()

## 📜 The Philosophy
OpenMLLab is born from the belief that to truly master Artificial Intelligence, one must move beyond `import sklearn`. This library is a documentation of a journey from raw mathematical equations found in research papers to optimized, vectorized Python implementations.

### Why Scratch?
* **Zero Abstraction:** Understand exactly how gradients flow through tensors.
* **Algorithmic Grit:** Solve numerical instability (Overflow/Underflow) without safety nets.
* **Scalable Systems:** Learn the architectural patterns used by industry legends.

---

## 🏗️ Library Architecture
We follow a modular, strictly-typed structure to ensure scalability and ease of contribution.

* `openmllab/linear_model/`: Optimized regression and classification (Logistics, Ridge, Lasso).
* `openmllab/metrics/`: Evaluation suite implementing standard industry benchmarks.
* `openmllab/preprocessing/`: Data normalization and transformation pipelines.
* `tests/`: Unit tests for mathematical correctness and shape-safety.

---

## 🚀 Easy-to-Use API
Designed to be intuitive for those familiar with the standard ML ecosystem.

```python
import numpy as np
from openmllab.metrics import accuracy_score

# Sample Data
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0])

# Evaluation
score = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {score * 100}%")

