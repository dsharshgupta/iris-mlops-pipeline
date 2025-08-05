import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

def evaluate_model(model_path='models/iris_model.pkl', scaler_path='models/scaler.pkl', data_path='data/iris.csv'):
    """Evaluates the model and logs metrics to an active MLflow run."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(data_path)
    
    X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    y = df['species']
            
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
            
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')

    if mlflow.active_run():
        mlflow.log_metric('full_dataset_accuracy', accuracy)
        mlflow.log_metric('full_dataset_precision', precision)
        mlflow.log_metric('full_dataset_recall', recall)
        mlflow.log_metric('full_dataset_f1_score', f1)

        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix on Full Dataset')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
            
        confusion_matrix_path = 'models/confusion_matrix.png'
        plt.savefig(confusion_matrix_path)
        plt.close()
        mlflow.log_artifact(confusion_matrix_path, "evaluation_results")

    evaluation_metrics = {'full_dataset_accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
    with open('models/evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)

    return evaluation_metrics