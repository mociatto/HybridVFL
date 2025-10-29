# HybridVFL: Disentangled Feature Learning for Edge-Enabled Vertical Federated Multimodal Classification

This repository contains four experiments implementing a hybrid vertical federated learning (HybridVFL) approach for skin lesion classification using both image and tabular clinical data.

## Experiments

### Experiment 1: Centralized Image-Only Baseline
[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/hybridvfl-exp-1-centralised-image-only)

Baseline experiment using only image data with a CNN architecture. Establishes the performance ceiling for image-only approaches.

### Experiment 2: Centralized Multimodal Model
[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/hybridvfl-exp-2-centralised-multimodal)

Enhanced model combining image features with clinical tabular data using cross-modal attention mechanisms. Demonstrates the benefit of multimodal fusion in centralized settings.

### Experiment 3: VFL Baseline
[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/hybridvfl-exp-3-simple-fusion-vfl)

Vertical federated learning implementation where image and tabular data are held by separate clients. Establishes VFL performance baseline without privacy-preserving constraints.

### Experiment 4: Cross-Modal VFL with Disentanglement
[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/mostafaanoosha/hybridvfl-exp-4-cross-modal-transformer-vfl)

Advanced VFL architecture with feature disentanglement and cross-modal transformer fusion. Demonstrates improved performance while maintaining federated learning principles.

## Required Datasets

Before running experiments, add these datasets to your Kaggle notebook:

- **Exp1 Dataset**: [hybridvfl-exp-1](https://www.kaggle.com/datasets/mostafaanoosha/hybridvfl-exp-1)
- **Exp2 Dataset**: [hybridvfl-exp-2](https://www.kaggle.com/datasets/mostafaanoosha/hybridvfl-exp-2)
- **Exp3 Dataset**: [hybridvfl-exp-3](https://www.kaggle.com/datasets/mostafaanoosha/hybridvfl-exp-3)

## Dataset Pipeline

Each experiment generates datasets that serve as inputs for subsequent experiments:

- **Experiment 1** → Creates `hybridvfl-exp-1` dataset (image features, models, labels)
- **Experiment 2** → Creates `hybridvfl-exp-2` dataset (tabular features, multimodal models)
- **Experiment 3** → Creates `hybridvfl-exp-3` dataset (VFL client models, results)
- **Experiment 4** → Uses all previous datasets for cross-modal VFL training

## Usage

1. Run experiments sequentially (Exp1 → Exp2 → Exp3 → Exp4)
2. Each experiment creates a Kaggle dataset for the next experiment
3. All datasets are also available in the `data/` folder for local execution

## Requirements

See `requirements.txt` for all required dependencies.

## Results

The experiments demonstrate progressive improvement from image-only (Exp1) to advanced cross-modal VFL (Exp4), showing the effectiveness of multimodal federated learning for medical image classification.
