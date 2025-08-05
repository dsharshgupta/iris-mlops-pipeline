import os
import pandas as pd
from src.data_loader import load_iris_data, poison_data
from src.train_model import train_iris_model
from src.evaluate_model import evaluate_model
import json
import mlflow

def run_experiment(poisoning_level):
    """
    Runs a single experiment with a given data poisoning level,
    logging all results to a dedicated MLflow run.
    """
    print(f"--- Running experiment with {poisoning_level*100}% data poisoning ---")

    # Set the experiment name
    mlflow.set_experiment("iris-poisoning-experiments")

    with mlflow.start_run(run_name=f"poisoning_{int(poisoning_level*100)}%"):
        # Log the poisoning level as a parameter
        mlflow.log_param("poisoning_level", poisoning_level)

        # Load and poison data
        df = load_iris_data()
        df_poisoned = poison_data(df.copy(), poisoning_level)
        poisoned_data_path = f'data/iris_poisoned_{int(poisoning_level*100)}.csv'
        df_poisoned.to_csv(poisoned_data_path, index=False)
        
        # Train model on poisoned data (metrics are logged within this function)
        model, scaler, training_metrics = train_iris_model(data_path=poisoned_data_path)

        # Evaluate model (metrics and artifacts are logged within this function)
        evaluation_metrics = evaluate_model(data_path=poisoned_data_path)

        # Store results locally as well
        results = {
            'poisoning_level': poisoning_level,
            'training_metrics': training_metrics,
            'evaluation_metrics': evaluation_metrics
        }
        
        output_file = f'models/results_{int(poisoning_level*100)}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Log the results file as an artifact
        mlflow.log_artifact(output_file, "results")

        print(f"--- Experiment for {poisoning_level*100}% poisoning complete. Run ID: {mlflow.active_run().info.run_id} ---")
        return results

if __name__ == "__main__":
    # Define poisoning levels, including a 0% baseline
    poisoning_levels = [0.0, 0.05, 0.10, 0.50]
    all_results = {}

    for level in poisoning_levels:
        result = run_experiment(level)
        all_results[f"{int(level*100)}%_poisoning"] = result

    # Save a summary of all experiments
    with open('models/all_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n--- All experiments complete ---")
    print("Results summary saved to models/all_experiment_results.json")
    print("Run 'mlflow ui' to view the results in the MLflow dashboard.")