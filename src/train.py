import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

def load_data(file_path):
    """
    Load processed dataset.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target_col="churn"):
    """
    Prepares features and target variable:
    - Drops target variable from features.
    - Splits data into training and testing sets.
    - Standardizes numerical features.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split data into train & test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def define_models():
    """
    Define different models for training and evaluation.
    """
    models = {
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    return models

def evaluate_model(model, X_test, y_test, name):
    """
    Evaluates the model using:
    - Classification report
    - Confusion matrix visualization
    """
    y_pred = model.predict(X_test)

    print(f"\n{name}:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    # Save the plot as an image
    image_path = f"confusion_matrix_{name}.png"
    plt.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate each model.
    """
    models = define_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test, name)

if __name__ == "__main__":
    # Load dataset
    file_path = os.path.join(os.getcwd().split("customer-churn-classification")[0], 'customer-churn-classification/data/processed/final_dataset.csv')
    df = load_data(file_path)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train and evaluate models
    train_and_evaluate_models(X_train, X_test, y_train, y_test)
    