# PaP-NF: Probabilistic Long-Term Time Series Forecasting via Prefix-as-Prompt Reprogramming and Normalizing Flows

[![Paper](https://img.shields.io/badge/Paper-Under_Review-orange)](#) 
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)

---

## 🌟 Overview

[cite_start]**PaP-NF** is a novel probabilistic forecasting framework that bridges the gap between Large Language Models (LLMs) and continuous time series data[cite: 7]. [cite_start]It utilizes a **Prefix-as-Prompt (PaP)** mechanism to align numerical features with a frozen LLM, effectively capturing global context without sacrificing numerical precision[cite: 27, 33].



### Key Features
- [cite_start]**Decoupled Design**: Separates global semantic reasoning from local numerical dynamics to prevent representational bottlenecks[cite: 27, 107].
- [cite_start]**Frozen LLM as Encoder**: Leverages the high-level reasoning of Llama-3.1 without the need for expensive fine-tuning[cite: 33, 190].
- [cite_start]**Normalizing Flow Decoder**: Employs conditional Planar Flows to generate flexible, multi-modal predictive distributions with exact likelihood estimation[cite: 53, 165].
- [cite_start]**Zero Tokenization Loss**: Bypasses discrete tokenization by using linear embeddings, preserving fine-grained numerical fidelity[cite: 31, 47].

---

## 🏗️ Architecture

[cite_start]The framework consists of three distinct stages[cite: 65, 108]:

1. [cite_start]**Stage 1: Local Pattern Encoding**: A linear encoder transforms partitioned input segments into a compact numerical representation $z$[cite: 93, 117].
2. **Stage 2: Global Context Extraction**: Learnable prefixes $P$ align $z$ with the LLM embedding space. [cite_start]A frozen Llama-3.1 model then extracts a global context vector $c$ via self-attention[cite: 94, 146, 149].
3. [cite_start]**Stage 3: Probabilistic Generation**: A Normalizing Flow decoder, conditioned on the fused features $h = Fuse(z, c)$, generates the final predictive distribution $p(Y|X)$[cite: 95, 160].



---

## 📊 Experimental Results

### 1. Long-Term Point Forecasting (MSE/MAE)
[cite_start]PaP-NF consistently achieves competitive results across major benchmarks, particularly excelling in long-horizon tasks[cite: 198].

| Dataset | Horizon ($H$) | TimesNet [23] | FEDformer [28] | **PaP-NF (Ours)** |
| :--- | :---: | :---: | :---: | :---: |
| **ETTh2** | 720 | 0.462 / 0.468 | 0.500 / 0.497 | [cite_start]**0.451 / 0.463** [cite: 209] |
| **ETTm2** | 720 | 0.408 / 0.403 | 0.421 / 0.415 | [cite_start]**0.395 / 0.391** [cite: 209] |
| **Traffic** | 720 | 0.640 / 0.350 | 0.626 / 0.382 | [cite_start]**0.618 / 0.337** [cite: 209] |

### 2. Probabilistic Accuracy (CRPS)
[cite_start]Evaluated at a 24-step horizon to assess distributional quality[cite: 192, 203].

| Model | ETTh1 | ETTh2 | Traffic |
| :--- | :---: | :---: | :---: |
| DeepAR [16] | 0.105 | 0.082 | **0.100** |
| **PaP-NF (Ours)** | **0.103** | **0.082** | 0.181 |
[cite_start]*(CRPS results from Table 2 [cite: 219])*

---

## 🔍 Qualitative Analysis



[cite_start]As shown in **Figure 4**, PaP-NF dynamically adjusts its uncertainty bands to cover challenging, high-volatility regions (top 10% absolute error steps of deterministic baselines), providing calibrated coverage for risk-sensitive applications[cite: 264, 265, 279].

---

## 🚀 Getting Started

### Installation
```bash
git clone [https://github.com/democracy04/PaP-NF.git](https://github.com/democracy04/PaP-NF.git)
cd PaP-NF
pip install -r requirements.txt
