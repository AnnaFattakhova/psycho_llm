# psycho_llm

A research repository for the study **“Is your LLM Agressive Narcissist with Depression? Analyzing Pathopsychological Portraits of Large Language Models”**.

This project investigates whether large language models (LLMs) demonstrate stable psychometric profiles when completing psychological questionnaires originally designed for humans. The repository contains experimental pipelines, prompts, result datasets and analysis scripts used to evaluate the psychological characteristics of contemporary LLMs.

## Overview

The study includes:

1) psychometric assessment of **19 LLMs**
2) evaluation using **11 psychological questionnaires**
3) comparison of:

  * API-based LLM responses
  * interface-based LLM responses
  * human responses
4) analysis of:

  * personality traits
  * attachment styles
  * temperament classifications
  * anxiety/depression indicators
  * aggression and dark personality traits
5) semantic clustering of model-generated explanations

## Repository Structure

```text
psycho_llm/
│
├── code/                # All the pipelines used in the study (Python)
├── prompts/             # Prompts used to generate responses for questionnaires
├── presentations/       # Project stages' description 
├── tests/               # Machine-readable questionnaire formats
└── README.md
```

## Questionnaires

The benchmark includes the following psychometric inventories:

* **BFI-2** — *Big Five Inventory–2*
* **16PF** — *16 Personality Factors Questionnaire* (Cattell’s 16 Personality Factors)
* **EPI** — *Eysenck Personality Inventory*
* **SD3** — *Short Dark Triad*
* **BDHI** — *Buss–Durkee Hostility Inventory*
* **ASQ** — *Attachment Style Questionnaire*
* **HADS** — *Hospital Anxiety and Depression Scale*
* **GAD-7** — *Generalized Anxiety Disorder-7*
* **BAI** — *Beck Anxiety Inventory*
* **BDI** — *Beck Depression Inventory*
* **SCL-90** — *Symptom Checklist-90-Revised*

## Experimental Design

The experiments consist of two major parts:

### 1. Psychometric profiling of LLMs

Models complete psychological questionnaires under controlled conditions:

* multiple temperatures
* repeated runs
* API-based access

### 2. Human–LLM correlation experiment

Human participants completed the same questionnaires and then asked the LLMs they regularly interact with to complete the tests. The study evaluates correlations between human and model responses.

## Methods

The repository includes implementations of:

* Pearson correlation analysis
* Fisher’s z-transformation
* leave-one-test-out sensitivity analysis
* systematic bias estimation
* MAE-based deviation analysis
* UMAP dimensionality reduction
* HDBSCAN clustering
* TF-IDF topic extraction

## License

This repository is released for research and academic purposes.
