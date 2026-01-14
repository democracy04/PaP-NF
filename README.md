# PaP-NF: Probabilistic Long-Term Time Series Forecasting via Prefix-as-Prompt Reprogramming and Normalizing Flows

[![Paper](https://img.shields.io/badge/Paper-Submitted_to_ICPR_2026-orange)](#) 
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)

---

## 🌟 Overview

[cite_start]**PaP-NF** is a novel probabilistic forecasting framework that aligns continuous time series with a frozen LLM using a **Prefix-as-Prompt (PaP)** mechanism[cite: 7]. [cite_start]It addresses the fundamental challenge of numerical precision loss in LLM-based approaches by decoupling global semantic extraction from local probabilistic generation[cite: 26, 27].

<p align="center">
  <img src="figures/overview__.png" width="850" alt="PaP-NF Architecture">
  <br>
  <em><b>Figure 1:</b> Overview of the PaP-NF framework. [cite_start]A linear encoder extracts local patterns, while the frozen LLM provides global semantic reasoning via prefix-based alignment[cite: 65, 93].</em>
</p>

### Key Innovations
* [cite_start]**Principled Hybrid Framework**: Preserves local numerical precision via linear embeddings while utilizing frozen LLMs for global reasoning[cite: 31].
* [cite_start]**Prefix-as-Prompt (PaP)**: Aligns numerical embeddings with pre-trained LLMs without modifying LLM parameters[cite: 33].
* [cite_start]**Uncertainty-Aware Prediction**: Conditions normalizing flows on joint numerical and LLM contexts for precise density estimation[cite: 34].
* [cite_start]**Efficient Sampling**: Achieves $O(1)$ efficiency in a single forward pass, bypassing the sampling latency of diffusion-based models[cite: 57, 257].

---

## 🏗️ Methodology

[cite_start]PaP-NF operates in three distinct stages[cite: 108]:
1. [cite_start]**Stage 1: Local Pattern Encoding**: Extracts localized temporal variations $z$ using an efficient linear TS encoder[cite: 70, 93, 110].
2. [cite_start]**Stage 2: Global Context Extraction**: Learnable prefixes $P$ align $z$ with the frozen LLM (Llama-3.1), producing a global context vector $c$ via average pooling[cite: 74, 94, 150].
3. [cite_start]**Stage 3: Probabilistic Future Generation**: A Normalizing Flow decoder generates the predictive distribution $p(Y|X)$ conditioned on the fused local and global features[cite: 89, 95, 157].


---

## 📊 Experimental Results

### 1. Long-Term Point Forecasting
[cite_start]PaP-NF consistently achieves competitive results across diverse benchmarks, particularly in long-horizon tasks where it reduces error accumulation[cite: 198, 201].

<p align="center">
  <img src="figures/table1.png" width="900" alt="Long-term Forecasting Results">
  <br>
  <em><b>Table 1:</b> Performance comparison on MSE/MAE. [cite_start]PaP-NF outperforms strong competitors like TimesNet on ETTh2 and ETTm2 for $H=720$[cite: 199, 208].</em>
</p>

### 2. Probabilistic Performance (CRPS)
[cite_start]The model provides robust uncertainty quantification, achieving state-of-the-art or second-best CRPS scores across various datasets[cite: 215, 216].

<p align="center">
  <img src="figures/table2.png" width="600" alt="CRPS Comparison">
  <br>
  [cite_start]<em><b>Table 2:</b> CRPS comparison against native probabilistic baselines ($H=24$)[cite: 218].</em>
</p>

---

## 🚀 Getting Started

### Installation
```bash
git clone [https://github.com/democracy04/PaP-NF.git](https://github.com/democracy04/PaP-NF.git)
cd PaP-NF
pip install -r requirements.txt
