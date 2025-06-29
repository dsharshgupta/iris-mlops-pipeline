import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json

def train_iris_model(data_path='data/iris.csv'):
    """Train a logistic regression model on IRIS data"""
    df = pd.read_csv(data_path)
    
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']
    X = df[feature_cols]
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/iris_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'test_size': len(y_test),
        'train_size': len(y_train)
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model trained with accuracy: {accuracy:.4f}")
    return model, scaler, metrics

if __name__ == "__main__":
    train_iris_model()
