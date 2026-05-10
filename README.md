# Federated Learning Research — Sparse Global Models with FedProx and Proximal L1 Regularization

## Overview

This repository contains research experiments exploring **Federated Learning (FL)** under **heterogeneous client data distributions**, with a focus on improving:

- Global model sparsity

The work builds on the **FedAvg** and **FedProx** frameworks and introduces a modified aggregation step using the **proximal operator of the L1 norm** to promote sparsity in the global model.

Our experiments were conducted primarily on the **FEMNIST dataset** using several notebook implementations and hyperparameter studies.

---

# Research Motivation

In many real-world applications such as:

- healthcare systems,
- personalized AI assistants,
- large language model personalization,
- mobile devices,

individual users often do not possess enough local data to train a strong machine learning model independently.

Federated Learning addresses this by allowing multiple clients to collaboratively train a shared model **without directly sharing their private data**.

However, FL introduces two major challenges:

## 1. Data Heterogeneity

Different clients may have drastically different data distributions.

Examples:
- handwriting styles differ between users,
- healthcare data differs across hospitals,
- user interaction patterns vary widely.

This often causes the global model to:
- generalize poorly,
- overfit dominant clients,
- struggle with convergence stability.

---

## 2. High-Dimensional Models

Modern neural networks contain millions of parameters, many of which may contribute very little to model performance.

This results in:
- increased communication cost,
- larger storage requirements,
- inefficient parameter usage.

---

# Proposed Approach

## FedProx for Heterogeneity

We build upon the **FedProx** algorithm, which modifies the local optimization objective by adding a proximal regularization term:

```math
F(x)=L(x)+\eta \|x-x_g\|^2
```

where:
- ```math\(L(x)\)``` is the local client loss,
- ```math\(x_g\)``` is the current global model,
- ```math\(\eta\)``` controls how strongly local models are constrained toward the global model.

### Intuition

FedProx reduces excessive client drift by discouraging local models from deviating too far from the shared global solution.

This improves:
- training stability,
- robustness under heterogeneous data,
- global model generalization.

---

## Promoting Sparsity with the Proximal Operator of the L1 Norm

To encourage sparse global models, we replace the standard aggregation update with the proximal operator of the L1 norm.

The proximal mapping performs soft-thresholding:

```math
T(x_i,\lambda)=\operatorname{sign}(x_i)\max(|x_i|-\lambda,0)
```

### Intuition

Parameters that are:
- close to zero → become exactly zero,
- large in magnitude → shrink toward zero.

This helps:
- eliminate non-informative parameters,
- reduce storage requirements,
- create more parameter-efficient models.

The proximal operator is applied during the global aggregation step after client updates are combined.

---

# Federated Learning Pipeline

The general workflow used in this project is:

1. Initialize a global model
2. Send the model to all participating clients
3. Train locally using SGD on each client dataset
4. Aggregate client updates on the server
5. Apply:
   - FedAvg, or
   - FedProx + proximal L1 regularization
6. Redistribute the updated global model
7. Repeat for multiple communication rounds

---

# Repository Structure

The repository contains several Jupyter notebooks corresponding to different experimental setups.

## Main Notebooks

- `FEMNIST - Fed_Avg_SGD.ipynb`
  - Baseline Federated Averaging implementation

- `FEMNIST - Fed_Avg_SGD Clean.ipynb`
  - Cleaned and organized FedAvg experiments

- `FEMNIST - Fed_Avg_SGD Clean 0 Start.ipynb`
  - Training initialized from zero-valued models

- `FEMNIST - Fed_Avg_SGD Clean 0 Start New Method.ipynb`
  - Implementation of proximal L1 regularization

Additional notebook variants contain hyperparameter tuning and experimental refinements.

---

# Dataset

## FEMNIST

Experiments were conducted on the FEMNIST dataset, a federated extension of MNIST that partitions handwritten character data across many users.

### Why FEMNIST?

FEMNIST naturally introduces:
- non-IID client distributions,
- client heterogeneity,
- realistic federated training conditions.

This makes it well-suited for evaluating:
- FedProx,
- sparsity-inducing techniques,
- federated optimization robustness.

---

# Experimental Goals

The experiments aim to evaluate:

- Global model accuracy
- Stability under heterogeneous client distributions
- Degree of vector sparsity
- Sensitivity to hyperparameters
- Tradeoffs between sparsity and performance

---

# Key Findings

## 1. Proximal L1 Regularization Promotes Sparsity

Applying the L1 proximal operator:
- significantly increased the number of zero-valued parameters,
- reduced parameter redundancy,
- improved model compactness.

---

## 2. Hyperparameter Sensitivity Matters

### L1 Regularization Parameter (\(\lambda\))

- Large \(\lambda\):
  - aggressively forces parameters to zero,
  - may prevent learning entirely.

- Small \(\lambda\):
  - preserves learning dynamics,
  - still promotes meaningful sparsity.

### FedProx Parameter (\(\eta\))

- Larger \(\eta\):
  - stronger pull toward the global model,
  - improves consistency.

- Smaller \(\eta\):
  - minimal effect on training.

---

# Applications

This research has potential applications in:

- Privacy-preserving healthcare AI
- Personalized language models
- Mobile and edge-device learning
- Continual federated model updates
- Communication-efficient distributed learning

Example:
A large language model could improve using private user interactions locally while only transmitting parameter updates instead of raw conversations.

---

# Acknowledgments

Special thanks to:

- **Yuyang Qiu**
- **Dr. Farzad Yousefian**
- Rutgers Mathematical Optimization Group

for their mentorship, guidance, and support throughout this research project.
