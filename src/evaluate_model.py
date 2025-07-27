import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # For Cloud Shell
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

def evaluate_model(model_path='models/iris_model.pkl', 
                  scaler_path='models/scaler.pkl',
                  data_path='data/iris.csv',
                  run_id_path='models/run_id.txt'): # **NEW:** Path to run ID file
    """Evaluate the trained model and log results to MLflow"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(data_path)
    
    # **NEW:** Read the run_id from the file
    try:
        with open(run_id_path, 'r') as f:
            run_id = f.read().strip()
    except FileNotFoundError:
        print("Error: run_id.txt not found. Cannot log to MLflow.")
        run_id = None

    # Only proceed with MLflow if we have a run_id
    if run_id:
        # Use the existing run to log evaluation metrics
        with mlflow.start_run(run_id=run_id):
            feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                           'petal length (cm)', 'petal width (cm)']
            X = df[feature_cols]
            y = df['species']
            
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            
            accuracy = accuracy_score(y, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
            
            # Log evaluation metrics to MLflow
            mlflow.log_metric('full_dataset_accuracy', accuracy)
            mlflow.log_metric('full_dataset_precision', precision)
            mlflow.log_metric('full_dataset_recall', recall)
            mlflow.log_metric('full_dataset_f1_score', f1)

            cm = confusion_matrix(y, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            # Save and log the confusion matrix as an artifact
            confusion_matrix_path = 'models/confusion_matrix.png'
            plt.savefig(confusion_matrix_path)
            plt.close()
            mlflow.log_artifact(confusion_matrix_path, "evaluation")
            
            print(f"Logged evaluation metrics to MLflow run: {run_id}")
    else:
        # Fallback for local execution without a training run
        # This part is optional but good practice
        X = df[feature_cols]
        y = df['species']
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)

    evaluation_metrics = {
        'full_dataset_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    with open('models/evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    print(f"Evaluation complete - Accuracy: {accuracy:.4f}")

    return evaluation_metrics

if __name__ == "__main__":
    evaluate_model()
