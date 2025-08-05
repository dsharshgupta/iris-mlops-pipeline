import os
import pandas as pd
from src.data_loader import load_iris_data, poison_data
from src.train_model import train_iris_model
from src.evaluate_model import evaluate_model
import json

def run_experiment(poisoning_level):
    """
    Runs a single experiment with a given data poisoning level.
    """
    print(f"--- Running experiment with {poisoning_level*100}% data poisoning ---")

    # Load and poison data
    df = load_iris_data()
    df_poisoned = poison_data(df.copy(), poisoning_level)
    poisoned_data_path = f'data/iris_poisoned_{int(poisoning_level*100)}.csv'
    df_poisoned.to_csv(poisoned_data_path, index=False)

    # Train model on poisoned data
    model, scaler, metrics = train_iris_model(data_path=poisoned_data_path)

    # Evaluate model
    evaluation_metrics = evaluate_model(data_path=poisoned_data_path)

    # Store results
    results = {
        'poisoning_level': poisoning_level,
        'training_metrics': metrics,
        'evaluation_metrics': evaluation_metrics
    }

    output_file = f'models/results_{int(poisoning_level*100)}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"--- Experiment for {poisoning_level*100}% poisoning complete. Results saved to {output_file} ---")
    return results

if __name__ == "__main__":
    poisoning_levels = [0.05, 0.10, 0.50]
    all_results = {}

    for level in poisoning_levels:
        result = run_experiment(level)
        all_results[f"{int(level*100)}%_poisoning"] = result

    # Also run with 0% poisoning for baseline
    baseline_result = run_experiment(0.0)
    all_results["0%_poisoning"] = baseline_result


    with open('models/all_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n--- All experiments complete ---")
    print("Results summary saved to models/all_experiment_results.json")