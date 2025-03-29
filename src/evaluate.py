import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd

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

def define_xgb_model():
    """Initialize an XGBoost classifier with base parameters."""
    return xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )


def get_hyperparameter_grid():
    """Return a dictionary containing hyperparameter grid for tuning."""
    return {
        'n_estimators': [100, 200, 500],  # Number of boosting rounds
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning rate
        'max_depth': [3, 5, 7, 10],  # Tree depth
        'subsample': [0.6, 0.8, 1.0],  # Row sampling
        'colsample_bytree': [0.6, 0.8, 1.0],  # Feature sampling
        'gamma': [0, 0.1, 0.2, 0.3],  # Minimum loss reduction
        'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight per child
        'reg_lambda': [0, 0.1, 1, 10],  # L2 Regularization
        'reg_alpha': [0, 0.1, 1, 10]  # L1 Regularization
    }


def perform_random_search(xgb_model, param_grid, X_train, y_train):
    """Perform RandomizedSearchCV to find the best hyperparameters."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=20,
        scoring='accuracy',
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1  # Utilize all available CPU cores
    )

    random_search.fit(X_train, y_train)

    print("Best Parameters:", random_search.best_params_)
    print("Best Cross-Validation Accuracy:", random_search.best_score_)

    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return classification metrics."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    
    #Saving model using Joblib
    import joblib
    joblib.dump(model, os.path.join(os.getcwd().split("customer-churn-classification")[0], 'customer-churn-classification/models/xgboost.pkl'))
    
    report['accuracy'] = accuracy
    return report


def save_metrics_to_json(report, filename=os.path.join(os.getcwd().split("customer-churn-classification")[0], 'customer-churn-classification/output/classification_metrics.json')):
    """Save evaluation metrics to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Model evaluation metrics saved to {filename}")
    
    
    


def explain_model_with_lime(model, X_train, X_test, feature_names, num_features=5, num_samples=500):
    """
    Generates LIME explanations for a model's predictions on test data.

    Parameters:
    - model: Trained model to explain (e.g., XGBoost, Random Forest, etc.).
    - X_train: Training data (used to fit LIME explainer).
    - X_test: Test data for which predictions need to be explained.
    - feature_names: List of feature names for proper explanation.
    - num_features: Number of features to display in the explanation.
    - num_samples: Number of samples to generate for local perturbations.

    Returns:
    - Dictionary of explanations for first few test instances.
    """
    
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),  
        feature_names=feature_names,  
        class_names=["No Churn", "Churn"],  
        mode="classification"
    )

    explanations = {}

    for i in range(3):  # Explain first 3 test instances (can be adjusted)
        exp = explainer.explain_instance(
            data_row=X_test_df.iloc[i],  
            predict_fn=model.predict_proba,  
            num_features=num_features,  
            num_samples=num_samples  
        )
        
        explanations[f"Instance {i+1}"] = exp.as_list()  # Store explanations as key-value pairs
        
        # Print explanation
        print(f"\nLIME Explanation for Test Instance {i+1}:")
        print(exp.as_list())
        exp.show_in_notebook()  # Visual representation (works in Jupyter)
        
    save_path = os.path.join(os.getcwd().split("customer-churn-classification")[0], 'customer-churn-classification/output/Lime_report.json')
    with open(save_path, "w") as f:
        json.dump(explanations, f, indent=4)

    print(f"\nLIME explanations saved to {save_path}")

    return explanations



if __name__ == "__main__":
    # Load dataset
    file_path = os.path.join(os.getcwd().split("customer-churn-classification")[0], 'customer-churn-classification/data/processed/final_dataset.csv')
    df = load_data(file_path)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    xgb_model = define_xgb_model()
    param_grid = get_hyperparameter_grid()

    # Perform hyperparameter tuning
    best_xgb = perform_random_search(xgb_model, param_grid, X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(best_xgb, X_test, y_test)
    
    save_metrics_to_json(metrics)
    
    X = df.drop(columns=["churn"])
    
    feature_names = X.columns.tolist()
    
    explain_model_with_lime(best_xgb, X_train, X_test, feature_names)