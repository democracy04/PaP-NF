# PaP-NF: Probabilistic Long-Term Time Series Forecasting via Prefix-as-Prompt Reprogramming and Normalizing Flows

[![Paper](https://img.shields.io/badge/Paper-Submitted_to_ICPR_2026-orange)](#) 
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)

---

## 🌟 Overview

**PaP-NF** is a novel probabilistic forecasting framework that aligns continuous time series with a frozen Large Language Model (LLM) using a **Prefix-as-Prompt (PaP)** mechanism. By utilizing the LLM as a global context encoder and integrating it with Normalizing Flows, the framework captures complex future uncertainties without the precision loss typically associated with numerical discretization.

<p align="center">
  <img src="figure/overview__.png" width="850">
</p>

### Key Innovations
* **Principled Hybrid Framework**: Maintains local numerical fidelity through linear embeddings while leveraging frozen LLMs for sophisticated global reasoning.
* **Prefix-as-Prompt (PaP)**: Successfully aligns numerical features with pre-trained LLM embedding spaces without the need for expensive parameter fine-tuning.
* **Uncertainty-Aware Forecasting**: Conditions conditional normalizing flows on joint numerical and semantic contexts for high-resolution density estimation.
* **High Efficiency**: Achieves $O(1)$ sampling speed in a single forward pass, significantly outperforming the iterative latency found in diffusion-based models.

---

## 🏗️ Methodology

Our framework consists of a three-stage pipeline designed for robust long-term prediction:
1. **Local Encoding**: A linear encoder captures localized temporal dynamics.
2. **Global Context**: Learnable prefixes align temporal features with a frozen Llama-3.1 backbone to extract semantic context.
3. **Probabilistic Generation**: Conditional Normalizing Flows generate exact likelihood-based future distributions $p(Y|X)$.

---

## 📊 Experimental Results

### 1. Competitive Point Forecasting
PaP-NF maintains superior accuracy across long-term horizons, notably outperforming state-of-the-art deterministic models like TimesNet on high-volatility datasets.

<p align="center">
  <img src="figure/table1.png" width="850">
</p>

### 2. Robust Uncertainty Quantification
The model provides well-calibrated predictive distributions, achieving top-tier performance in Continuous Ranked Probability Score (CRPS) benchmarks.

<p align="center">
  <img src="figure/table2.png" width="600">
</p>

---

## 🚀 Getting Started

### Installation
```bash
git clone [https://github.com/democracy04/PaP-NF.git](https://github.com/democracy04/PaP-NF.git)
cd PaP-NF
pip install -r requirements.txt
