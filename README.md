# PaP-NF: Probabilistic Long-Term Time Series Forecasting via Prefix-as-Prompt Reprogramming and Normalizing Flows

[![Paper](https://img.shields.io/badge/Paper-Submitted_to_ICPR_2026-orange)](#) 
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)

---

## 🌟 Overview

[cite_start]**PaP-NF** is a novel probabilistic forecasting framework that aligns continuous time series with a frozen Large Language Model (LLM) using a **Prefix-as-Prompt (PaP)** mechanism[cite: 7]. [cite_start]It addresses the fundamental challenge of numerical precision loss in LLM-based approaches by decoupling global semantic extraction from local probabilistic generation[cite: 26, 27].

<p align="center">
  <img src="figures/overview__.png" width="850">
  <br>
  <em><b>Figure 1:</b> Overview of the PaP-NF framework. [cite_start]A linear encoder extracts local patterns, while the frozen LLM provides global semantic reasoning via prefix-based alignment[cite: 65, 93].</em>
</p>

### Key Innovations
* [cite_start]**Principled Hybrid Framework**: Preserves local numerical precision via linear embeddings while utilizing frozen LLMs for global semantic reasoning[cite: 31].
* [cite_start]**Prefix-as-Prompt (PaP)**: Aligns numerical embeddings with pre-trained LLMs without modifying LLM parameters[cite: 33].
* [cite_start]**Uncertainty-Aware Prediction**: Conditions normalizing flows on joint numerical and LLM contexts for precise density estimation[cite: 34].
* [cite_start]**Efficient Sampling**: Achieves $O(1)$ efficiency in a single forward pass, bypassing the sampling latency of diffusion-based models[cite: 57, 257].

---

## 🏗️ Methodology

The PaP-NF framework operates in three distinct stages:
1. [cite_start]**Stage 1 (Local Pattern Encoding)**: Extracts localized temporal features $z$ using an efficient linear TS encoder[cite: 71, 93, 108].
2. [cite_start]**Stage 2 (Global Context Extraction)**: Learnable prefixes $P$ align $z$ with a frozen Llama-3.1 model to extract a global context vector $c$ via average pooling[cite: 94, 102, 108].
3. [cite_start]**Stage 3 (Probabilistic Generation)**: Conditions a Normalizing Flow (Planar Flows) on the fused representation $h = Fuse(z, c)$ to generate the predictive distribution $p(Y|X)$[cite: 88, 95, 108].

---

## 📊 Experimental Results

### 1. Long-Term Point Forecasting (MSE/MAE)
[cite_start]PaP-NF achieves competitive results across diverse benchmarks, maintaining robust performance even as the prediction horizon increases[cite: 198].

<p align="center">
  <img src="figures/table1.png" width="850">
  <br>
  <em><b>Table 1:</b> Long-term forecasting performance comparison. [cite_start]PaP-NF outperforms strong competitors like TimesNet on ETTh2 and ETTm2 at $H=720$[cite: 199, 208].</em>
</p>

### 2. Probabilistic Performance (CRPS)
[cite_start]PaP-NF provides high-quality predictive distributions, capturing multi-modal uncertainty robustly across all datasets[cite: 9, 215].

<p align="center">
  <img src="figures/table2.png" width="600">
  <br>
  [cite_start]<em><b>Table 2:</b> CRPS comparison against native probabilistic baselines ($H=24$)[cite: 218, 219].</em>
</p>

---

## 🚀 Getting Started

### Installation
```bash
git clone [https://github.com/democracy04/PaP-NF.git](https://github.com/democracy04/PaP-NF.git)
cd PaP-NF
pip install -r requirements.txt
