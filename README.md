# PaP-NF: Probabilistic Long-Term Time Series Forecasting via Prefix-as-Prompt Reprogramming and Normalizing Flows

<p align="center">
  <img src="https://img.shields.io/badge/Paper-Submitted_to_ICPR_2026-orange">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/github/stars/democracy04/PaP-NF?style=social">
</p>
[📄 Paper](./paper.pdf)

---

## 🌟 Overview

**PaP-NF** is a novel probabilistic forecasting framework that aligns continuous time series with a frozen Large Language Model (LLM) using a **Prefix-as-Prompt (PaP)** mechanism. By utilizing the LLM as a global context encoder and integrating it with Normalizing Flows, the framework captures complex future uncertainties without the precision loss typically associated with numerical discretization.

<p align="center">
  <img src="figure/overview__.png" width="80%">
</p>

### Key Innovations
* **Principled Hybrid Framework**: Maintains local numerical fidelity through linear embeddings while leveraging frozen LLMs for sophisticated global reasoning[cite: 31].
* **Prefix-as-Prompt (PaP)**: Aligns numerical embeddings with pre-trained LLMs without modifying backbone parameters[cite: 33].
* **Uncertainty-Aware Forecasting**: Conditions normalizing flows on joint numerical and semantic contexts for precise density estimation[cite: 34].
* **High Efficiency**: Achieves $O(1)$ sampling speed, bypassing the iterative latency of diffusion-based models[cite: 57, 257].

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
  <img src="figure/table1.png" width="65%">
</p>

### 2. Robust Uncertainty Quantification
The model provides well-calibrated predictive distributions, achieving top-tier performance in Continuous Ranked Probability Score (CRPS) benchmarks.

<p align="center">
  <img src="figure/table2.png" width="75%">
</p>

---
  
# Citation
  @article{kim2026papnf,
  title={PaP-NF: Probabilistic Long-Term Time Series Forecasting via Prefix-as-Prompt Reprogramming and Normalizing Flows},
  author={Kim, Minju and Hur, Youngbum},
  journal={Submitted to International Conference on Pattern Recognition (ICPR)},
  year={2026}
}

