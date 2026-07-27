# FAST-FedNano: Privacy-Preserving Federated Learning for Intelligent Nanoparticle Drug Delivery

## Overview

FAST-FedNano is an AI framework that uses privacy-preserving Federated Learning to predict the results of nanoparticle drug formulations without having to disclose sensitive pharmaceutical data from the involved institutions.

It explores whether it is possible for several simulated pharmaceutical labs (hospitals) to train a good quality machine learning algorithm by using their data without violating their privacy through Federated Learning (FL) and Differential Privacy (DP).

The FAST-FedNano framework is a combination of centralized machine learning, federated learning, privacy-preserving optimization, and multi-objective formulation optimization.

The system integrates:

* Federated Learning with Flower (FedAvg)
* Differential Privacy with Opacus (DP-SGD)
* Machine Learning with Scikit-learn, XGBoost, and PyTorch
* Multi-objective Optimization with NSGA-II (pymoo)
* Cloudflare Worker to host the model
* HTML, CSS, JavaScript-based Prediction Interface

The framework allows researchers to:

* Predict nanoparticle encapsulation efficiency
* Study the performance of centralized and federated learning
* Explore the privacy and utility tradeoff
* Optimize nanoparticle formulations with artificial intelligence
* Deploy models via a web app

---

# Objectives

* Develop a privacy preserving machine learning system for nanoparticle formulation prediction
* Evaluate the performance of centralized and federated learning
* Protect the client’s data with the use of Differential Privacy
* Develop simulation Non-IID pharmaceutical institutions
* Optimize nanoparticle formulations using multi-objective optimization
* Deploy the trained models using a light-weight inference web application

---

# Key Features

## Centralized Machine Learning

* Data preprocessing pipeline
* Random Forest
* XGBoost
* Multi-Layer Perceptron (MLP)
* Hyperparameter tuning using RandomizedSearchCV
* Cross-validation performance evaluation

---

## Federated Learning

* Flower (FedAvg)
* Five simulated pharmaceutical clients
* Non-IID data partitioning
* Local training with weighted aggregation
* Communication round evaluation

---

## Differential Privacy

* DP-SGD using Opacus
* Configurable privacy budget (ε)
* Privacy–utility trade-off analysis
* Secure local model updates

---

## Multi-objective Optimization

* NSGA-II optimization
* Pareto-optimal formulation discovery
* Simultaneous optimization of:

  * Encapsulation Efficiency
  * Loading Capacity
  * Particle Size

---

## Web Deployment

* Cloudflare Workers
* Lightweight inference API
* Interactive web interface
* Portable JSON model bundle

---

## Model Comparison

Performance comparison between:

* Centralized models
* Federated models
* Differentially Private Federated models

---

# Technologies Used

| Category             | Tools / Libraries     |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Machine Learning     | Scikit-learn          |
| Gradient Boosting    | XGBoost               |
| Deep Learning        | PyTorch               |
| Federated Learning   | Flower                |
| Differential Privacy | Opacus                |
| Optimization         | pymoo (NSGA-II)       |
| Data Processing      | Pandas, NumPy         |
| Visualization        | Matplotlib            |
| Model Serialization  | Joblib                |
| Deployment           | Cloudflare Workers    |
| Frontend             | HTML, CSS, JavaScript |

---

# Project Architecture

## 1. Data Preparation

* Load nanoparticle formulation dataset
* Missing value imputation
* Feature scaling
* Train-test split

---

## 2. Centralized Learning

Train multiple machine learning models:

* Random Forest
* XGBoost
* Neural Network (MLP)

Evaluate baseline performance.

---

## 3. Non-IID Client Partitioning

* Simulate five pharmaceutical laboratories
* Dirichlet distribution partitioning
* Realistic heterogeneous datasets

---

## 4. Federated Learning

Each client:

* Trains local model
* Sends model weights
* FedAvg aggregates weights
* Updates global model

---

## 5. Differential Privacy

Local training uses:

* DP-SGD
* Gradient clipping
* Noise injection
* Privacy budget tracking

---

## 6. Multi-objective Optimization

NSGA-II searches formulation parameters to:

* Maximize Encapsulation Efficiency
* Maximize Loading Capacity
* Minimize Particle Size

---

## 7. Deployment

Export trained models into

```
model_bundle.json
```

Deploy using

* Cloudflare Worker
* Interactive Web UI

---

# Dataset

**Dataset Source**

Mendeley PLGA Nanoparticle Formulation Dataset

Dataset contains:

* 433 experimental samples
* 15 numerical features

Target variables:

* Particle Size
* Encapsulation Efficiency (Primary Target)
* Loading Capacity

Feature categories:

Drug descriptors

* Molecular Weight
* LogP
* TPSA
* Melting Point
* Hydrogen Bond Acceptors
* Hydrogen Bond Donors

Formulation parameters

* Polymer Molecular Weight
* LA/GA Ratio
* Drug/Polymer Ratio
* Surfactant Concentration
* HLB
* pH
* Solvent Polarity

---

# Experimental Pipeline

## Phase 1

Centralized Machine Learning

* Data preprocessing
* Feature scaling
* Hyperparameter tuning
* Model comparison

---

## Phase 2

Non-IID Client Simulation

* Dirichlet partitioning
* Five clients
* Heterogeneous data distribution

---

## Phase 3

In-Silico Simulation Engine

* Synthetic formulation generation
* Dataset augmentation

---

## Phase 4

Federated Learning

* Flower
* FedAvg
* Global model aggregation

---

## Phase 5

Differential Privacy

* DP-SGD
* Privacy budget evaluation
* Accuracy comparison

---

## Phase 6

Formulation Optimization

* NSGA-II
* Pareto front generation
* Recommended nanoparticle formulations

---

# Results Summary

| Model              | Training Strategy | Test R²                  |
| ------------------ | ----------------- | ------------------------ |
| XGBoost            | Centralized       | **0.638**                |
| Random Forest      | Centralized       | **0.631**                |
| Federated MLP      | FedAvg            | **0.467**                |
| Centralized MLP    | Centralized       | **0.450**                |
| Federated MLP + DP | DP-SGD            | Privacy-Utility Analysis |

Key observations:

* Tree-based methods are better than neural networks for small table-like datasets.
* Federated Learning performs equally well without requiring data to be shared.
* Differential Privacy offers good privacy assurances without compromising on accuracy too much.
* Federated averaging slightly boosts the generalization ability of neural networks.

---

# Repository Structure

```
FAST-FedNano/

├── data_preprocessing.py
├── train_model_improved.py
├── partition_clients.py
├── federated_learning.py
├── federated_dp_opacus.py
├── phase6_nsga2_optimization.py
├── export_model.py
├── predict_new.py
├── predict_batch.py
├── clients/
├── federated_outputs/
├── phase6_outputs/
├── ml-worker/
├── frontend/
├── README.md
```

---

# How to Run

## Clone Repository

```bash
git clone https://github.com/yourusername/FAST-FedNano.git

cd FAST-FedNano
```

---

## Install Dependencies

```bash
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run Centralized Training

```bash
python data_preprocessing.py

python train_model_improved.py
```

---

## Partition Clients

```bash
python partition_clients.py
```

---

## Run Federated Learning

```bash
python federated_learning.py
```

---

## Run Differential Privacy Training

```bash
python federated_dp_opacus.py
```

---

## Run Optimization

```bash
python phase6_nsga2_optimization.py
```

---

## Export Models

```bash
python export_model.py
```

---

## Deploy Cloudflare Worker

```bash
cd ml-worker

wrangler deploy
```

---

## Run Frontend

```bash
cd frontend

python -m http.server 8000
```

Open

```
http://localhost:8000
```

---

# Future Scope

* Real-world hospital integration
* Secure Aggregation
* Homomorphic Encryption
* Transformer-based prediction models
* Federated XGBoost
* Explainable AI (SHAP/LIME)
* Cloud-native deployment
* Integration with pharmaceutical laboratories
* Automated hyperparameter optimization
* Digital Twin simulation for nanoparticle formulation

---

# Applications

* AI-assisted Drug Delivery
* Pharmaceutical Research
* Federated Healthcare AI
* Privacy-Preserving Machine Learning
* Drug Formulation Optimization
* Academic Research
* B.Tech / M.Tech Thesis
* AI-based Decision Support Systems

---

# References

* McMahan et al. (2017) — Federated Learning (FedAvg)
* Abadi et al. (2016) — Differential Privacy
* Dwork (2006) — Differential Privacy
* Hsu et al. (2019) — Non-IID Federated Learning
* Grinsztajn et al. (2022) — Tree Models vs Deep Learning

---

# License

This project is licensed under the MIT License.

---

# Author

**Developed by Harsh Arora**

