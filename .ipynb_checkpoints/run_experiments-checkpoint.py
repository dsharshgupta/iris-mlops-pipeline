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

    # Set the experiment name for all poisoning runs
    mlflow.set_experiment("iris-poisoning-experiments")

    # Start a new MLflow run for this specific poisoning level
    with mlflow.start_run(run_name=f"poisoning_{int(poisoning_level*100)}%"):
        # Log the poisoning level as a parameter for easy filtering in the UI
        mlflow.log_param("poisoning_level", poisoning_level)

        # Load original data and create a poisoned version
        df = load_iris_data()
        df_poisoned = poison_data(df.copy(), poisoning_level)
        poisoned_data_path = f'data/iris_poisoned_{int(poisoning_level*100)}.csv'
        df_poisoned.to_csv(poisoned_data_path, index=False)
        
        # Train the model on the (potentially) poisoned data
        # The train_model function will log its own metrics to our active run
        model, scaler, training_metrics = train_iris_model(data_path=poisoned_data_path)

        # Evaluate the model on the full (poisoned) dataset
        # The evaluate_model function will also log its metrics and artifacts
        evaluation_metrics = evaluate_model(data_path=poisoned_data_path)

        # Print the final accuracy to the console log for GitHub Actions
        final_accuracy = evaluation_metrics.get('full_dataset_accuracy', 0)
        print(f"Poisoning Level: {poisoning_level*100}%, Full Dataset Accuracy: {final_accuracy:.4f}")

        # Store a summary of the results locally
        results = {
            'poisoning_level': poisoning_level,
            'training_metrics': training_metrics,
            'evaluation_metrics': evaluation_metrics
        }
        
        output_file = f'models/results_{int(poisoning_level*100)}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Log the detailed JSON results as an artifact in MLflow
        mlflow.log_artifact(output_file, "results")

        print(f"--- Experiment for {poisoning_level*100}% poisoning complete. MLflow Run ID: {mlflow.active_run().info.run_id} ---")
        return results

if __name__ == "__main__":
    # Define the poisoning levels to test, including a 0% baseline
    poisoning_levels = [0.0, 0.05, 0.10, 0.50]
    all_results = {}

    # Run the experiment for each level
    for level in poisoning_levels:
        result = run_experiment(level)
        all_results[f"{int(level*100)}%_poisoning"] = result

    # Save a final summary of all experiments to a single file
    with open('models/all_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n--- All experiments complete ---")
    print("Results summary saved to models/all_experiment_results.json")
    print("You can now view the detailed results in the MLflow UI by running 'mlflow ui'")