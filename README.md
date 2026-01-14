# PaP-NF: Probabilistic Long-Term Time Series Forecasting via Prefix-as-Prompt Reprogramming and Normalizing Flows

[![Paper](https://img.shields.io/badge/Paper-Submitted_to_ICPR_2026-orange)](#) 
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)

---

## 🌟 Overview

[cite_start]**PaP-NF**는 **Prefix-as-Prompt (PaP)** 메커니즘을 통해 시계열 데이터를 Frozen LLM에 정렬하는 새로운 확률론적 예측 프레임워크입니다[cite: 7]. [cite_start]LLM을 글로벌 컨텍스트 인코더로 활용하고 Normalizing Flow를 결합하여, 수치적 정밀도 손실 없이 복잡한 미래의 불확실성을 효과적으로 캡처합니다[cite: 26, 27].

<p align="center">
  <img src="figures/overview__.png" width="850">
  <br>
  <em><b>Figure 1:</b> PaP-NF 프레임워크 개요. [cite_start]Linear Encoder가 국부 패턴을 추출하고, Frozen LLM이 PaP 정렬을 통해 글로벌 의미론적 추론을 수행합니다[cite: 65, 93].</em>
</p>

### Key Innovations
* [cite_start]**Principled Hybrid Framework**: 선형 임베딩을 통해 로컬 수치 정밀도를 유지하면서 Frozen LLM의 강력한 추론 능력을 결합합니다[cite: 31].
* [cite_start]**Prefix-as-Prompt (PaP)**: LLM 파라미터 수정 없이 수치 임베딩을 LLM 공간에 정렬하는 학습 가능한 접두사 메커니즘을 도입합니다[cite: 33].
* [cite_start]**Uncertainty-Aware Prediction**: 수치적 특징과 LLM 컨텍스트를 Normalizing Flow에 조건화하여 정밀한 밀도 추정을 수행합니다[cite: 34].
* [cite_start]**Efficient Sampling**: Diffusion 모델과 달리 단일 패스로 $O(1)$의 샘플링 효율성을 달성합니다[cite: 57, 257].

---

## 📊 Experimental Results

### 1. Long-Term Point Forecasting (MSE/MAE)
[cite_start]PaP-NF는 장기 예측 벤치마크에서 기존 SOTA 모델인 TimesNet 대비 우수한 성능을 보여줍니다[cite: 199]. 특히 ETTh2 및 ETTm2 데이터셋에서 높은 정확도를 유지합니다.

<p align="center">
  <img src="figures/table1.png" width="850">
  <br>
  [cite_start]<em><b>Table 1:</b> 주요 베이스라인과의 장기 예측 성능 비교 ($H \in \{96, 192, 336, 720\}$)[cite: 208].</em>
</p>

### 2. Probabilistic Performance (CRPS)
[cite_start]확률론적 예측 정확도를 측정하는 CRPS 지표에서도 경쟁력 있는 성능을 입증하였습니다[cite: 215, 219].

<p align="center">
  <img src="figures/table2.png" width="600">
  <br>
  [cite_start]<em><b>Table 2:</b> 다양한 확률론적 예측 모델과의 CRPS 비교 ($H=24$)[cite: 218].</em>
</p>

---

## 🚀 Getting Started

### Installation
```bash
git clone [https://github.com/democracy04/PaP-NF.git](https://github.com/democracy04/PaP-NF.git)
cd PaP-NF
pip install -r requirements.txt
