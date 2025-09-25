Project Overview

This project implements Air Quality Index (AQI) forecasting using a hybrid model:

An enhanced Extreme Learning Machine (ELM / KELM) whose hyperparameters (hidden nodes, weights, thresholds) are optimized via a Genetic Algorithm (GA)

A Bi-directional LSTM (BiLSTM) model (or extension) to further refine predictions

(Optionally) a combined/ensemble approach, comparing among GA-KELM, BiLSTM, or hybrid versions

The goal is to produce more accurate forecasts of AQI (or pollutant concentrations) than baseline models. The methodology is inspired by research such as “Air Quality Index Forecasting via Genetic Algorithm-Based Improved Extreme Learning Machine (GA-KELM)” 
OpenReview
+2
jnao-nu.com
+2
, and similar works combining GA, ELM, and BiLSTM as comparative/extension models 
IJISRT
+2
IJPREMS
+2
.
AQI_Forecasting/
│  
├── data/
│   ├── raw_data.csv            ← original pollutant / meteorological data  
│   ├── processed_data.csv      ← cleaned & feature-engineered data  
│   └── train_test_split.pkl    ← pickled train/test splits  
│  
├── src/
│   ├── preprocess.py           ← data cleaning, normalization, feature engineering  
│   ├── ga_kel_model.py         ← implementation of GA + ELM / KELM  
│   ├── bilstm_model.py         ← BiLSTM model definitions & training  
│   ├── hybrid_model.py         ← code to combine GA-KELM + BiLSTM if used  
│   ├── evaluate.py             ← metrics (RMSE, MAE, MSE, etc.)  
│   └── utils.py                ← helper functions (e.g. saving/loading, plotting)  
│  
├── experiments/
│   ├── config_ga_kel.yaml      ← hyperparameter settings for GA-KELM  
│   ├── config_bilstm.yaml      ← hyperparameter settings for BiLSTM  
│   └── results/                ← logs, output predictions, metric files  
│  
├── requirements.txt            ← Python library dependencies  
├── run_ga_kel.py               ← script to run GA-KELM workflow end to end  
├── run_bilstm.py               ← script to train & test BiLSTM  
├── run_hybrid.py               ← script to run the hybrid / ensemble approach  
└── README.md                    ← documentation (this file)  
Required Libraries & Environment Setup

In your requirements.txt, list all the third-party Python packages needed, for example:

numpy
pandas
scikit-learn
tensorflow
keras
torch            # if using PyTorch LSTM
matplotlib
seaborn
deap             # or another library for genetic algorithm
joblib
pickle
yaml             # for config files


You may also include versions, e.g.:

numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
tensorflow>=2.4
keras>=2.4
deap>=2.0.0

Installation Steps

In your README you can instruct:

# (Optional) create a virtual environment
python3 -m venv venv
source venv/bin/activate      # on Linux/macOS
venv\Scripts\activate         # on Windows

# Install dependencies
pip install -r requirements.txt


Also mention if GPU / CUDA is required (for LSTM training). If using TensorFlow / PyTorch, you may need tensorflow-gpu or proper GPU drivers.

How to Run / Execution Steps

Below is a general sequence of steps. You may provide example commands in your README so others can reproduce your results.

Preprocessing

python src/preprocess.py --input data/raw_data.csv --output data/processed_data.csv


This cleans, handles missing values, normalizes / scales features, does feature engineering (lag features, rolling statistics), and splits into train/test sets (saving a pickle or CSV).

Run GA-KELM model

python run_ga_kel.py --config experiments/config_ga_kel.yaml


This script should read the processed data, read hyperparameter ranges from config, run the genetic algorithm to find optimal ELM / KELM parameters, train on training set, test on test set, and output predictions + metrics into experiments/results/.

Run BiLSTM model

python run_bilstm.py --config experiments/config_bilstm.yaml


This script builds the BiLSTM (or variants), trains it, evaluates on the test set, and outputs predictions + metrics.

Run Hybrid / Ensemble (if applicable)

python run_hybrid.py --config experiments/config_hybrid.yaml


This can combine predictions (e.g. stacking, voting, weighted average) from GA-KELM and BiLSTM or cascade one into another.

Evaluate & Plot
Use src/evaluate.py or utilities in utils.py to compute metrics (MSE, RMSE, MAE, R², etc.) and generate plots (actual vs predicted, residual plots, loss curves).

In your README, provide sample commands (with any required flags) so that users know exactly what to run and in what order.

Key Algorithmic Parts — What Each File Does

Here are more details you can include, so readers understand the internals:

preprocess.py

Load raw dataset (pollutants, meteorological variables)

Handle missing or invalid values (e.g. interpolation, forward fill)

Generate lag features (e.g. pollutant_t−1, pollutant_t−2) and/or rolling means

Normalize or standardize features (e.g. MinMaxScaler or StandardScaler)

Split into training and test sets (e.g. 70/30 or 80/20)

Save splits (e.g. using pickle or CSV)

ga_kel_model.py

Define the ELM / KELM model: single hidden layer feedforward network (or kernel version), randomly assign input weights and biases, compute hidden layer output, solve output weights via least squares

Define kernel extension (KELM) where kernel functions are used in place of hidden layer computations

Define Genetic Algorithm:

Encode candidate solutions (e.g. number of hidden nodes, weights, biases or kernel parameter values)

Define fitness function (e.g. RMSE on validation set)

Provide selection, crossover, mutation operators

Iterate over generations to find best candidate

Train final ELM / KELM with best parameters and test on unseen data

Save the model and predictions

bilstm_model.py

Build a Bi-directional LSTM neural network (e.g. one or more BiLSTM layers, dropout, dense layers)

Define loss (e.g. MSE) and optimizer (e.g. Adam)

Train for certain epochs, with validation split

Save best model (via checkpoint)

Predict on test set and return metrics + predictions

hybrid_model.py (optional)

Logic to combine outputs from GA-KELM and BiLSTM, e.g.:

Weighted averaging of predictions

Using one model’s output as input to the other

Stacking (meta-learner)

Produce final combined predictions and metrics

evaluate.py

Functions to compute MSE, RMSE, MAE, R², etc.

Functions to plot actual vs predicted, residual errors, error histograms, training/validation loss curves

utils.py

Generic helper functions: logging, model saving/loading, plotting utilities, config parsing, etc.

run_*.py

Entry scripts that glue preprocessing, model training, and evaluation together

They parse command-line arguments (e.g. --config), load configs, call appropriate modules in src/, store results in experiments/

config_*.yaml

YAML or JSON files specifying hyperparameter ranges / defaults, e.g. for GA: population size, mutation rate, crossover rate, number of generations; for BiLSTM: number of layers, hidden size, learning rate, batch size, epochs

Example README Section Template

Below is a suggested snippet for your README:

📦 Installation & Setup
# Clone repository
git clone https://github.com/yourusername/AQI_Forecasting.git
cd AQI_Forecasting

# (Optionally) create and activate virtual environment
python3 -m venv venv
source venv/bin/activate    # (or venv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt

🧠 How to Run

Preprocess the data

python src/preprocess.py --input data/raw_data.csv --output data/processed_data.csv


Train / test the GA-KELM model

python run_ga_kel.py --config experiments/config_ga_kel.yaml


Train / test the BiLSTM model

python run_bilstm.py --config experiments/config_bilstm.yaml


(Optional) Run hybrid combination

python run_hybrid.py --config experiments/config_hybrid.yaml


Inspect results

Output predictions and metrics are stored in experiments/results/

Use src/evaluate.py or plotting scripts to visualize performance

📁 File & Module Breakdown
File	Purpose
data/	Raw & processed data files
src/preprocess.py	Data cleaning, feature creation, train/test split
src/ga_kel_model.py	GA + ELM / KELM implementation
src/bilstm_model.py	BiLSTM model definition & training
src/hybrid_model.py	Logic for combining models (if used)
src/evaluate.py	Metric computation and evaluation
src/utils.py	Helper functions
run_ga_kel.py, run_bilstm.py, run_hybrid.py	Entry scripts to run workflows
experiments/config_*.yaml	Hyperparameter settings for experiments
experiments/results/	Output logs, predictions, metrics, plots
🧩 Notes on the Methods

The GA-KELM approach uses a Genetic Algorithm to optimize hyperparameters of the Extreme Learning Machine / Kernel Extreme Learning Machine: hidden node counts, weights, biases, or kernel parameters. The objective (fitness) is often defined via RMSE or MSE on validation data. 
jnao-nu.com
+2
IJPREMS
+2

The BiLSTM model captures temporal dependencies in both forward and backward directions. Many papers combine BiLSTM with GA-KELM (or compare them) to demonstrate improved accuracy. 
IJISRT
+2
IJPREMS
+2

In hybrid approaches, predictions from both models (or their intermediate outputs) may be fused via stacking or averaging to achieve better generalization.

✅ Tips & Troubleshooting

Make sure your data has no missing or invalid values after preprocessing.

Scale / normalize inputs appropriately — ELM & LSTM are sensitive to input scaling.

For GA, choose reasonable population size and number of generations so optimization completes in acceptable time.

Use GPU (if available) to accelerate BiLSTM training.

Seed your random number generators (NumPy, TensorFlow / PyTorch, GA library) for reproducibility.

Log training/validation loss curves to monitor overfitting.
