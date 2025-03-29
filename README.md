# Churn Prediction Model

## Overview
This repository provides a **churn prediction pipeline** using machine learning models like **Random Forest** and **XGBoost**. It includes data preprocessing, feature engineering, model training, and evaluation.

## Setup

### Install Dependencies  
Ensure you have Python installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

## Install Dependencies  
```bash
├── data/                  # Raw and processed data
│   ├── processed/
├── src/
│   ├── data_preprocessing/
│       ├── preprocess.py           
│       ├── feature_engineering.py  
├── ├── training/
│       ├── Evaluate_model.py           
│       ├── train_model.py  
├── reports/
│   ├── classification.json
│   ├── feature_importance.png
├── models/
│   ├── best_xgb_model.pkl
├── README.md              # Project documentation
├── requirements.txt       # Required dependencies
├── results.csv
```

## Features Explanation  

- **Plan_type**: The type of subscription plan the customer has chosen.  
- **Customer_time_period**: The number of days since the customer's first transaction.  
- **Plan_type_premium**: A binary indicator (1 or 0) showing if the customer has a premium plan.  
- **Plan_type_standard**: A binary indicator (1 or 0) showing if the customer has a standard plan.  
- **Stock_return**: The percentage change in the stock market over a given period.  
- **Stock_volatility**: Measures the fluctuation in stock market trends over a rolling period.  


## Usage

### Preprocess Data

```bash
python src/data_preprocessing/preprocess.py
```
### Feature Engineering

```bash
python src/data_preprocessing/feature_engineering.py
```
### Train the model

```bash
python src/train.py
```
### Final finetuning and evaluation
```bash
python src/evaluate.py
```
## Model Explanation using Lime

LIME (Local Interpretable Model-agnostic Explanations) is a powerful tool to interpret black-box models by explaining individual predictions. 
Advantages of LIME

* Simple & Intuitive: Breaks down model predictions into feature contributions.

+ Works on Any Model: Compatible with XGBoost, Random Forest, Neural Networks, etc.

- Feature Importance Clarity: Highlights key features influencing each prediction.

## Results

The trained models generate performance metrics like accuracy, precision, recall, and F1-score. Results are logged and visualized through confusion matrices.

