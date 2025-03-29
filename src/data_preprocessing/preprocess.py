import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os

"""
# Data Preprocessing Pipeline for Customer Churn Classification

This script performs preprocessing on a customer churn dataset, including:
1. Loading Data: Reads the dataset from a CSV file.
2. Date Conversion: Converts specified columns (`date`, `issuing_date`) to datetime format.
3. Handling Missing Values: 
   - Fills missing `transaction_amount` with the median.
   - Fills missing `plan_type` with the most frequent category.
   - Applies one-hot encoding to `plan_type`.
4. Encoding Categorical Features: 
   - Label encodes categorical columns for machine learning compatibility.
5. Imputing Missing/Infinite Values:
   - Replaces infinite values with NaN.
   - Imputes missing values using the mean.
6. Final Processing:
   - Drops the `churn` column (if present) to prepare features for modeling.
   - Saves two preprocessed datasets:
     - `preprocess_part1.csv`: Fully processed data.
     - `preprocess_part2.csv`: Data with only date and missing values handled.
"""


def load_data(file_path):
    """
    Load dataset from the given file path.
    """
    return pd.read_csv(file_path)

def preprocess_dates(df, date_columns):
    """
    Convert specified date columns to datetime format.
    """
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    return df

def handle_missing_values(df):
    """
    Handle missing values:
    - Fill missing transaction amounts with the median.
    - Fill missing categorical values (plan_type) with the most frequent value.
    """
    df["transaction_amount"].fillna(df["transaction_amount"].median(), inplace=True)
    df["plan_type"].fillna(df["plan_type"].mode()[0], inplace=True)
    
    # One-hot encoding for 'plan_type'
    df = pd.get_dummies(df, columns=["plan_type"], drop_first=True)
    
    return df

def encode_categorical_features(df):
    """
    Encode categorical features:
    - One-hot encode 'plan_type' and drop the first category to avoid multicollinearity.
    - Label encode other categorical columns.
    """    

    # Label encode other categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store label encoders for inverse transformation later

    return df, label_encoders

def impute_missing_values(df):
    """
    Handle missing values and infinite values:
    - Replace infinite values with NaN.
    - Impute NaN values using the mean.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy="mean")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df

def preprocess_data(file_path):
    """
    Complete preprocessing pipeline:
    - Load data
    - Convert date columns
    - Handle missing values
    - Encode categorical features
    - Impute missing/infinite values
    """
    df = load_data(file_path)
    df1 = df.copy()
    df1 = preprocess_dates(df1, ["date", "issuing_date"])
    df1 = handle_missing_values(df1)
    
    df, label_encoders = encode_categorical_features(df)
    df = impute_missing_values(df)

    # Drop the target column 'churn' (if preparing features for training)
    df.drop(columns=["churn"], inplace=True, errors="ignore")
    
    df.to_csv(os.path.join(os.getcwd().split("customer-churn-classification")[0], 'customer-churn-classification/data/processed/preprocess_part1.csv'))
    df1.to_csv(os.path.join(os.getcwd().split("customer-churn-classification")[0], 'customer-churn-classification/data/processed/preprocess_part2.csv'))
    

    return df, label_encoders

if __name__ == "__main__":
    file_path = os.path.join(os.getcwd().split("customer-churn-classification")[0], 'customer-churn-classification/data/Original/churn_data.csv')
    df_preprocessed, encoders = preprocess_data(file_path)
    print("Data preprocessing completed successfully!")
