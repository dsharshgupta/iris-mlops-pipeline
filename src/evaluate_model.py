import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # For Cloud Shell
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path='models/iris_model.pkl', 
                  scaler_path='models/scaler.pkl',
                  data_path='data/iris.csv'):
    """Evaluate the trained model"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(data_path)
    
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']
    X = df[feature_cols]
    y = df['species']
    
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
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
