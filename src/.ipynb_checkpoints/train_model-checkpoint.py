import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json
import mlflow
import mlflow.sklearn

def train_iris_model(data_path='data/iris.csv'):
    """
    Train a logistic regression model on IRIS data, ensuring all artifacts are
    saved locally and logged to MLflow if a run is active.
    """
    df = pd.read_csv(data_path)
    
    params = {
        'random_state': 42,
        'max_iter': 200,
        'test_size': 0.2
    }
    
    # --- Feature Engineering and Model Training ---
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)', 'location']
    X = df[feature_cols]
    y = df['species']
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=params['test_size'], random_state=params['random_state'], stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=params['random_state'], max_iter=params['max_iter'])
    model.fit(X_train_scaled, y_train)
    
    # --- Metrics Calculation ---
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # --- Save Artifacts Locally (Guaranteed) ---
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/iris_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoder, 'models/label_encoder.pkl')
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report
    }
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # --- Log to MLflow (If Active) ---
    if not mlflow.active_run():
        mlflow.start_run()

    print("Logging to active MLflow run...")
    mlflow.log_params(params)
    mlflow.log_metric('training_accuracy', accuracy)
    mlflow.log_metric('training_f1_weighted', report['weighted avg']['f1-score'])
    
    mlflow.sklearn.log_model(model, "iris_model")
    mlflow.log_artifact('models/scaler.pkl', "preprocessing")
    mlflow.log_artifact('models/label_encoder.pkl', "preprocessing")
    mlflow.log_artifact('models/metrics.json', "training_reports")

    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    mlflow.end_run()

    return model, scaler, metrics

if __name__ == '__main__':
    train_iris_model()