import os
import pytest
import pandas as pd
import numpy as np
from src.train_model import train_iris_model
from src.data_loader import load_iris_data

@pytest.fixture(autouse=True)
def setup_mlflow_for_tests(monkeypatch, tmpdir):
    """
    Fixture to configure MLflow to use a local backend for testing.
    This runs automatically for every test in this file.
    """
    mlruns_dir = tmpdir.mkdir("mlruns")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{mlruns_dir}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", f"{mlruns_dir}/artifacts")
    yield
    monkeypatch.undo()

def test_model_accuracy():
    """Test model accuracy is within a reasonable range."""
    if not os.path.exists('data/iris.csv'):
        load_iris_data()
    
    model, scaler, metrics = train_iris_model()
    
    assert 'accuracy' in metrics
    assert metrics['accuracy'] > 0.85

def test_model_files_exist():
    """Test if model, scaler, and encoder files are created locally."""
    # Ensure data exists before training
    if not os.path.exists('data/iris.csv'):
        load_iris_data()

    # Run the training process which should create the files
    train_iris_model()
    
    # Assert that all expected files were created in the 'models' directory
    assert os.path.exists('models/iris_model.pkl'), "Model file was not created"
    assert os.path.exists('models/scaler.pkl'), "Scaler file was not created"
    assert os.path.exists('models/metrics.json'), "Metrics file was not created"
    assert os.path.exists('models/label_encoder.pkl'), "Label encoder file was not created"

def test_sanity_check():
    """Sanity test to ensure the model can make predictions."""
    if not os.path.exists('data/iris.csv'):
        load_iris_data()

    model, scaler, _ = train_iris_model()
    
    # Create a sample DataFrame for prediction
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2, 0]], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'location'])
    
    scaled_sample = scaler.transform(sample)
    prediction = model.predict(scaled_sample)
    
    assert prediction is not None
    # The model now predicts an integer because of the LabelEncoder
    assert isinstance(prediction[0], np.int64)
