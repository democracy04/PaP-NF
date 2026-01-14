# PaP-NF: Probabilistic Long-Term Time Series Forecasting via Prefix-as-Prompt Reprogramming and Normalizing Flows

[![Paper](https://img.shields.io/badge/Paper-Submitted_to_ICPR_2026-orange)](#) 
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)

---

## 🌟 Overview

**PaP-NF** is a novel probabilistic forecasting framework that aligns continuous time series with a frozen Large Language Model (LLM) using a **Prefix-as-Prompt (PaP)** mechanism[cite: 7]. [cite_start]By utilizing the LLM as a global context encoder and integrating it with Normalizing Flows, the framework captures complex future uncertainties without the precision loss typically associated with numerical discretization[cite: 26, 27].

<p align="center">
  <img src="figure/overview__.png" width="80%">
</p>

### Key Innovations
* [cite_start]**Principled Hybrid Framework**: Preserves local numerical precision via linear embeddings while utilizing frozen LLMs for global semantic reasoning[cite: 31].
* [cite_start]**Prefix-as-Prompt (PaP)**: Aligns numerical embeddings with pre-trained LLMs without modifying LLM parameters[cite: 33].
* [cite_start]**Uncertainty-Aware Prediction**: Conditions normalizing flows on joint numerical and LLM contexts for precise density estimation[cite: 34].
* [cite_start]**Efficient Sampling**: Achieves $O(1)$ efficiency in a single forward pass, bypassing the sampling latency of diffusion-based models[cite: 57, 257].

---

## 🏗️ Methodology

Our framework consists of a three-stage pipeline designed for robust long-term prediction:
1. [cite_start]**Local Encoding**: A linear encoder captures localized temporal dynamics[cite: 93].
2. [cite_start]**Global Context**: Learnable prefixes align temporal features with a frozen Llama-3.1 backbone to extract semantic context[cite: 94].
3. [cite_start]**Probabilistic Generation**: Conditional Normalizing Flows generate exact likelihood-based future distributions $p(Y|X)$[cite: 95].

---

## 📊 Experimental Results

### 1. Competitive Point Forecasting
[cite_start]PaP-NF maintains superior accuracy across long-term horizons, notably outperforming state-of-the-art deterministic models like TimesNet on high-volatility datasets[cite: 199].

<p align="center">
  <img src="figure/table1.png" width="75%">
</p>

### 2. Robust Uncertainty Quantification
[cite_start]The model provides well-calibrated predictive distributions, achieving top-tier performance in Continuous Ranked Probability Score (CRPS) benchmarks[cite: 215].

<p align="center">
  <img src="figure/table2.png" width="55%">
</p>

---

## 🚀 Getting Started

### Installation
```bash
git clone [https://github.com/democracy04/PaP-NF.git](https://github.com/democracy04/PaP-NF.git)
cd PaP-NF
pip install -r requirements.txt
