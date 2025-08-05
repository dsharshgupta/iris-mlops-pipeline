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
    """Train a logistic regression model on IRIS data and log metrics and artifacts to MLflow if a run is active."""
    df = pd.read_csv(data_path)
    
    params = {
        'random_state': 42,
        'max_iter': 200,
        'test_size': 0.2
    }
    
    # Check if there is an active MLflow run
    if mlflow.active_run():
        mlflow.log_params(params)
    
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']
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
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log metrics and artifacts only if in an active MLflow run
    if mlflow.active_run():
        mlflow.log_metric('training_accuracy', accuracy)
        mlflow.log_metric('training_precision_weighted', report['weighted avg']['precision'])
        mlflow.log_metric('training_recall_weighted', report['weighted avg']['recall'])
        mlflow.log_metric('training_f1_weighted', report['weighted avg']['f1-score'])
        
        # Log the trained model
        mlflow.sklearn.log_model(model, "iris_model")
        
        # Save and log the scaler and encoder as artifacts
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(encoder, 'models/label_encoder.pkl')
        mlflow.log_artifact('models/scaler.pkl', "preprocessing")
        mlflow.log_artifact('models/label_encoder.pkl', "preprocessing")

    # Always save model files locally
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
    
    print(f"Model trained with accuracy on test split: {accuracy:.4f}")
        
    return model, scaler, metrics

if __name__ == "__main__":
    # This part is for running the script standalone
    mlflow.set_experiment("iris-classification-cml")
    with mlflow.start_run():
        train_iris_model()