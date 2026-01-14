# PaP-NF: Probabilistic Long-Term Time Series Forecasting via Prefix-as-Prompt Reprogramming and Normalizing Flows
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)]() 
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)]()

---

## Overview

**PaP-NF** is a probabilistic long-term forecasting framework that:

- Uses **Prefix-as-Prompt (PaP)** to align continuous time series into an LLM embedding space
- Employs a **frozen LLM** purely as a global context encoder
- Generates flexible **probabilistic future trajectories** via **conditional Normalizing Flows**

This hybrid design avoids the precision loss of tokenization and bypasses diffusion-style iterative sampling  
while maintaining competitive long-term point accuracy.

---

## Key Features

- Hybrid deterministic + probabilistic architecture
- Frozen LLM as global semantic extractor
- Linear numerical encoder for stable local patterns
- Normalizing Flow decoder enabling multi-modal outputs
- Fast O(1) sampling (vs. diffusion’s O(T))
- Prefix-based reprogramming without modifying LLM weights

---

## Installation (Planned)

```bash
git clone https://github.com/<username>/PaP-NF.git
cd PaP-NF

pip install -r requirements.txt
