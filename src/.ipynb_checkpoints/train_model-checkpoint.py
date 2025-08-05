import pandas as pd
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
    """Trains a model and logs metrics to an active MLflow run."""
    df = pd.read_csv(data_path)
    params = {'random_state': 42, 'max_iter': 200, 'test_size': 0.2}

    if mlflow.active_run():
        mlflow.log_params(params)

    X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    y = df['species']
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=params['test_size'], random_state=params['random_state'], stratify=y_encoded)
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(random_state=params['random_state'], max_iter=params['max_iter']).fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    if mlflow.active_run():
        mlflow.log_metric('training_accuracy', accuracy)
        mlflow.log_metric('training_f1_weighted', report['weighted avg']['f1-score'])
        mlflow.sklearn.log_model(model, "iris_model")
        
        joblib.dump(scaler, 'models/scaler.pkl')
        mlflow.log_artifact('models/scaler.pkl', "preprocessing")
        joblib.dump(encoder, 'models/label_encoder.pkl')
        mlflow.log_artifact('models/label_encoder.pkl', "preprocessing")

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/iris_model.pkl')
    
    metrics = {'accuracy': accuracy, 'classification_report': report}
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    return model, scaler, metrics