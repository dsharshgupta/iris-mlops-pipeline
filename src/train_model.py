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
    """Train a logistic regression model on IRIS data and optionally log with MLflow"""
    df = pd.read_csv(data_path)
    
    params = {
        'random_state': 42,
        'max_iter': 200,
        'test_size': 0.2
    }
    
    run_id = None
    
    # Check if there is an active MLflow run
    if mlflow.active_run():
        run_id = mlflow.active_run().info.run_id
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
    
    # Log metrics and artifacts only if in an MLflow run
    if mlflow.active_run():
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision_weighted', report['weighted avg']['precision'])
        mlflow.log_metric('recall_weighted', report['weighted avg']['recall'])
        mlflow.log_metric('f1_weighted', report['weighted avg']['f1-score'])
        mlflow.sklearn.log_model(model, "iris_model")
        scaler_path = joblib.dump(scaler, 'models/scaler.pkl')[0]
        mlflow.log_artifact(scaler_path, "scaler")

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
    if run_id:
        print(f"MLflow Run ID: {run_id}")
        with open('models/run_id.txt', 'w') as f:
            f.write(run_id)
        
    return model, scaler, metrics

if __name__ == "__main__":
    mlflow.set_experiment("iris-classification-cml")
    with mlflow.start_run():
        train_iris_model()
