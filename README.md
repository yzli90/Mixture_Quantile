# Mixture quantile framework

A flexible python framework for **Mixture quantile (MQ) models** . This project supports complex basis function generation and constrained optimization.

## Project structure

* **Main.py**: The primary entry point to run standard training, fitting, and evaluation tasks, which coordinates the workflow by calling **cross_validation**.
* **Models.py**: Contains the core logic for the **Mixture quantile** model along with GMM and GPD benchmarks. It includes essential methods for **fit**, **predict**, and **density** calculation.
* **Basis_generators.py**: Implementation of different basis functions (e.g., I-splines, distributional basis) used to construct the **Mixture quantile** curves.
* **Metrics.py**: Specialized diagnostic tools for evaluating performance, featuring customized risk and accuracy measures.
* **Cross_validation.py**: Logic for K-fold validation to ensure both model stability and performance robustness.
* **Report.py**: Utility functions for generating diagnostic plots and performance tables for result analysis.

## Installation

1. **Environment**: Python 3.11+ is recommended.
2. **Dependencies**:
   ```bash
   pip install -r Requirements.txt
