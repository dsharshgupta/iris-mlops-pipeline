import os
import pytest
import pandas as pd
from src.train_model import train_iris_model
from src.data_loader import load_iris_data

# Use a pytest fixture to set up a temporary local MLflow tracking URI for all tests in this file
@pytest.fixture(autouse=True)
def setup_mlflow_for_tests(monkeypatch, tmpdir):
    """
    Fixture to configure MLflow to use a local backend for testing,
    preventing attempts to connect to a remote server.
    """
    # Create a temporary directory for mlflow runs
    mlruns_dir = tmpdir.mkdir("mlruns")
    
    # Use monkeypatch to set environment variables for the duration of the tests
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{mlruns_dir}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", f"{mlruns_dir}/artifacts")
    
    # Yield control back to the test function
    yield
    
    # Teardown (optional, as tmpdir is handled by pytest)
    monkeypatch.undo()


def test_model_accuracy():
    """Test model accuracy is within a reasonable range"""
    if not os.path.exists('data/iris.csv'):
        load_iris_data()
    
    # The training function will now use the local MLflow URI set by the fixture
    model, scaler, metrics = train_iris_model()
    
    assert 'accuracy' in metrics
    # A simple sanity check for the Iris dataset
    assert metrics['accuracy'] > 0.85


def test_model_files_exist():
    """Test if model and scaler files are created"""
    if not os.path.exists('data/iris.csv'):
        load_iris_data()

    train_iris_model()
    
    assert os.path.exists('models/iris_model.pkl')
    assert os.path.exists('models/scaler.pkl')
    assert os.path.exists('models/metrics.json')


def test_sanity_check():
    """Sanity test to ensure the model can make predictions"""
    if not os.path.exists('data/iris.csv'):
        load_iris_data()

    model, scaler, _ = train_iris_model()
    
    # Create a dummy sample for prediction
    # This corresponds to ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    
    scaled_sample = scaler.transform(sample)
    prediction = model.predict(scaled_sample)
    
    assert prediction is not None
    assert isinstance(prediction[0], str) # Assuming the output is the species name
