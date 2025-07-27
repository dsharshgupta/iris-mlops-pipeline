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
    """
    mlruns_dir = tmpdir.mkdir("mlruns")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{mlruns_dir}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", f"{mlruns_dir}/artifacts")
    yield
    monkeypatch.undo()


def test_model_accuracy():
    """Test model accuracy is within a reasonable range"""
    if not os.path.exists('data/iris.csv'):
        load_iris_data()
    
    model, scaler, metrics = train_iris_model()
    
    assert 'accuracy' in metrics
    assert metrics['accuracy'] > 0.85


def test_model_files_exist():
    """Test if model, scaler, and encoder files are created"""
    if not os.path.exists('data/iris.csv'):
        load_iris_data()

    train_iris_model()
    
    assert os.path.exists('models/iris_model.pkl')
    assert os.path.exists('models/scaler.pkl')
    assert os.path.exists('models/metrics.json')
    # **NEW:** Check for the label encoder file
    assert os.path.exists('models/label_encoder.pkl')


def test_sanity_check():
    """Sanity test to ensure the model can make predictions"""
    if not os.path.exists('data/iris.csv'):
        load_iris_data()

    model, scaler, _ = train_iris_model()
    
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    
    scaled_sample = scaler.transform(sample)
    prediction = model.predict(scaled_sample)
    
    assert prediction is not None
    # **FIX:** The model now predicts an integer because of the LabelEncoder
    assert isinstance(prediction[0], np.int64)
