import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import shap
from fairlearn.metrics import MetricFrame, selection_rate
from mlflow.exceptions import MlflowException

def evaluate_model(model_path='models/iris_model.pkl', scaler_path='models/scaler.pkl', data_path='data/iris.csv'):
    """Evaluates the model and logs metrics to an active MLflow run."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(data_path)
    
    X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'location']]
    y = df['species']
    location_sensitive = df['location']
            
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
            
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')

    try:
        mlflow.set_experiment("iris-fairness-experiment")
        with mlflow.start_run():
            mlflow.log_metric('full_dataset_accuracy', accuracy)
            mlflow.log_metric('full_dataset_precision', precision)
            mlflow.log_metric('full_dataset_recall', recall)
            mlflow.log_metric('full_dataset_f1_score', f1)

            # --- Fairlearn Part ---
            metrics = {
                'accuracy': accuracy_score,
                'recall_per_class': lambda y_true, y_pred: recall_score(y_true, y_pred, average=None, zero_division=0),
                'selection_rate': selection_rate
            }
            grouped_on_location = MetricFrame(metrics=metrics,
                                              y_true=y,
                                              y_pred=y_pred,
                                              sensitive_features=location_sensitive)

            print("\n--- Fairlearn Metrics (Grouped by Location) ---")
            print(grouped_on_location.by_group)
            
            mlflow.log_dict(grouped_on_location.by_group.to_dict(), "fairness_metrics_by_location.json")
            # --- End of Fairlearn Part ---

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

            # SHAP explainer
            X_summary = shap.kmeans(X_scaled, 10)
            explainer = shap.KernelExplainer(model.predict_proba, X_summary)
            shap_values = explainer.shap_values(X_scaled)

            shap.summary_plot(shap_values[2], X_scaled, show=False)
            plt.title('SHAP Summary Plot for Virginica')
            shap_plot_path = 'models/shap_summary_plot_virginica.png'
            plt.savefig(shap_plot_path)
            plt.close()
            mlflow.log_artifact(shap_plot_path, "evaluation_results")

    except MlflowException as e:
        print(f"WARNING: Could not connect to MLflow server. Skipping logging. Error: {e}")

    evaluation_metrics = {'full_dataset_accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
    with open('models/evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)

    return evaluation_metrics

if __name__ == '__main__':
    evaluate_model()
