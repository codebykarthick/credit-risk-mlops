# 🏦 Credit Risk Classification Pipeline
This project implements a complete machine learning pipeline for credit risk prediction using tabular data. It leverages gradient boosting models (e.g. XGBoost, LightGBM), automated experiment tracking (MLflow), and model interpretability tools (SHAP). The goal is to build a robust, reproducible workflow for classifying whether a person is likely to default on credit payments within the next two years.

---

## 📂 Dataset

We use the **[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)** dataset from Kaggle.

To use this project, download the following training data file from the [competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it in a `data/raw/` directory:

- `cs-training.csv` (main training data which is split into train, val and test sets for our evaluation)

> ⚠️ **Note**: The data file is not included in this repository. You must manually download it from Kaggle.

---

## 🛠️ Setup
A `Makefile` is included for running the critical flows, please check it for usage. It supports a variety of reproducible commands.

1. **Clone the repository**:

```bash
git clone git@github.com:codebykarthick/credit-risk-mlops.git
cd credit-risk-mlops
```

2. **Create a virtual environment and install dependencies**:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. **Prepare the data**:
* Download the csv training file from kaggle and place them in `data/raw` folder in the project root.

4. **Run commands**:
Check `scripts/run.sh` for sample commands or use the makefile (all runs everything from first to last).

## Structure
* `notebooks/`: Contains the Jupyter notebooks used for the simple exploration of the data.
* `src/`: All the code used for preprocessing, feature engineering, training, evaluation and serving the trained model.

## 📈 Goals
* Build an explainable credit default classifier
* Use MLflow for experiment tracking and model registry
* Optimize model hyperparameters using Optuna
* Explain model predictions using SHAP
* Deploy the model via FastAPI

## 🚧 Status
This project is a work in progress. The current focus is on setting up the base pipeline and training loop.

