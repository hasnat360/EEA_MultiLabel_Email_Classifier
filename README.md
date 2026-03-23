# EEA MultiLabel Email Classifier

This repository contains the codebase for the Continuous Assessment (CA1) of the Engineering and Evaluating Artificial Intelligence module.

## Modular Email Classification Architecture

The architecture is designed to classify incoming support emails into multiple dependent variables (Type 2, Type 3, and Type 4) using two distinct architectural design choices:
1. **Chained Multi-Output Classification (DC1):** Sequentially combines dependent variables into deeper targets (e.g., `y2`, `y2 | y3`, `y2 | y3 | y4`) and trains one Multi-Class classifier per chain level. 
2. **Hierarchical Modelling (DC2):** Uses a parent model's predictions to route test samples to branch-specific subsequent classifiers along a hierarchy (`y2` -> `y3` -> `y4`).

This codebase rigorously adheres to separation of concerns, providing isolated modules for data preprocessing, feature extraction (TF-IDF), train-test encapsulation, and abstract modeling interfaces.

## Project Structure
- `Code/Config.py`: Centralized configuration for targets, chains, file paths, and hyperparameters.
- `Code/preprocess.py`: Loads raw datasets, handles deduplication, noise removal, and translates text if configured.
- `Code/embeddings.py`: Computes TF-IDF matrices from ticket components.
- `Code/data_model.py`: Wraps training data structures (`Data` and `FilteredData`) for uniform injection into classifiers.
- `Code/model/`: Contains `base.py` (Abstract Base Class) and `randomforest.py` (concrete classifier).
- `Code/modelling.py`: Contains and manages the architecture-specific logic for DC1 and DC2 runs.
- `Code/main.py`: The entry point script that connects all components and generates the comparative output.
- `Report/`: Contains the academic evaluation report for this submission.

## Installation & Execution

### Requirements
- Python 3.8+
- `pandas>=1.5`
- `scikit-learn>=1.2`
- `numpy>=1.23`

### Run Instructions
Navigate to the `Code/` directory to run the pipeline:
```bash
cd Code
pip install -r requirements.txt
python main.py
```
*Note: Ensure the `Code/data/` directory contains `AppGallery.csv` and `Purchasing.csv` before running.*