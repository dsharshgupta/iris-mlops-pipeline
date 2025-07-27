import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json
import mlflow
import mlflow.sklearn

def train_iris_model(data_path='data/iris.csv'):
    """Train a logistic regression model on IRIS data and log with MLflow"""
    df = pd.read_csv(data_path)
    
    # Define model parameters
    params = {
        'random_state': 42,
        'max_iter': 200,
        'test_size': 0.2
    }
    
    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(params)
        
        feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                       'petal length (cm)', 'petal width (cm)']
        X = df[feature_cols]
        y = df['species']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params['test_size'], random_state=params['random_state'], stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=params['random_state'], max_iter=params['max_iter'])
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision_weighted', report['weighted avg']['precision'])
        mlflow.log_metric('recall_weighted', report['weighted avg']['recall'])
        mlflow.log_metric('f1_weighted', report['weighted avg']['f1-score'])

        # Log model and scaler as artifacts
        mlflow.sklearn.log_model(model, "iris_model")
        mlflow.log_artifact(joblib.dump(scaler, 'models/scaler.pkl')[0], "scaler")

        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/iris_model.pkl')
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'test_size': len(y_test),
            'train_size': len(y_train)
        }
        
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")
        
    return model, scaler, metrics

if __name__ == "__main__":
    # Set MLflow experiment
    # The tracking URI will be set by the environment variable in the CI/CD pipeline
    mlflow.set_experiment("iris-classification-cml")
    train_iris_model()
