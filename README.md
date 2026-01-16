# PaP-NF: Probabilistic Long-Term Time Series Forecasting via Prefix-as-Prompt Reprogramming and Normalizing Flows

<p align="center">
  <img src="https://img.shields.io/badge/Paper-Submitted_to_ICPR_2026-orange">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/github/stars/democracy04/PaP-NF?style=social">
</p>
<p align="center">
  <a href="./paper.pdf"> [Paper]</a> 
</p>


## 🌟 Overview

**PaP-NF** is a probabilistic forecasting framework that aligns continuous time series with a frozen Large Language Model (LLM) using a **Prefix-as-Prompt (PaP)** mechanism. By utilizing the LLM as a global context encoder and integrating it with Normalizing Flows, the framework captures complex future uncertainties without the precision loss typically associated with numerical discretization.

<p align="center">
  <img src="figure/Overview_.png" width="90%">
</p>

### Key Innovations
* **Principled Hybrid Framework**: Maintains local numerical fidelity through linear embeddings while leveraging frozen LLMs for sophisticated global reasoning.
* **Prefix-as-Prompt (PaP)**: Aligns numerical embeddings with pre-trained LLMs without modifying backbone parameters.
* **Uncertainty-Aware Forecasting**: Conditions normalizing flows on joint numerical and semantic contexts for precise density estimation.
* **High Efficiency**: Achieves $O(1)$ sampling speed, bypassing the iterative latency of diffusion-based models.

---

## 💾 Datasets

The following benchmark datasets used in the paper can be obtained from the links below:

* **ETT (Electricity Transformer Temperature)**: Available at [ETDataset GitHub](https://github.com/zhouhaoyi/ETDataset). Includes ETTh1, ETTh2, ETTm1, and ETTm2.
* **Traffic**: Standard transportation dataset often hosted in [Time-Series-Library](https://github.com/thuml/Time-Series-Library).

You can start training and evaluation by running main.py. Ensure your datasets are in the ./data/ folder.

```
PaP-NF/
 └─ data/
     ├─ ETTh1.csv
     ├─ ETTh2.csv
     ├─ ETTm1.csv
     ├─ ETTm2.csv
     └─ Traffic.csv
```
---

## 🚀 Getting Started

### 1. Environment Setup
Clone the repository and install the required packages.

```bash
git clone https://github.com/democracy04/PaP-NF.git
cd PaP-NF
pip install -r requirements.txt
```

### 2. Dataset & Config

Update dataset location inside the `Config` class in `main.py`:

```python
class Config:
    seq_len = 336
    pred_len = 96
    prefix_len = 16
    num_flows = 8
    root_path = './dataset/numeric'
    data_path = 'ETTm2.csv'
    llm_model_name = 'meta-llama/Meta-Llama-3.1-8B'
```

To use a different dataset, place the file in your chosen directory  
and update `root_path` and `data_path` accordingly.


### 3. Train & Evaluate

The current `main.py` automatically runs experiments for multiple horizons in a single pass:

```python
if __name__ == '__main__':
    horizons = [96, 192, 336, 720]
    for pred_len in horizons:
        ...
```

To launch the full experiment (all horizons) with the default configuration:

```bash
python main.py
```
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
  <img src="figure/table2.png" width="65%">
</p>

---

## 📁 Project Structure

```
PaP-NF/
 ├─ data/
 ├─ data_loader.py
 ├─ metric.py
 ├─ main.py
 └─ requirements.txt
```


