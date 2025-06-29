import pytest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from train_model import train_iris_model
from evaluate_model import evaluate_model

def test_model_accuracy():
    """Test model accuracy"""
    if not os.path.exists('data/iris.csv'):
        from data_loader import load_iris_data
        load_iris_data()
    
    model, scaler, metrics = train_iris_model()
    assert metrics['accuracy'] > 0.9

def test_model_files_exist():
    """Test if model files are created"""
    if not os.path.exists('data/iris.csv'):
        from data_loader import load_iris_data
        load_iris_data()
    
    train_iris_model()
    assert os.path.exists('models/iris_model.pkl')
    assert os.path.exists('models/scaler.pkl')
    assert os.path.exists('models/metrics.json')

def test_sanity_check():
    """Sanity test"""
    if not os.path.exists('data/iris.csv'):
        from data_loader import load_iris_data
        load_iris_data()
    
    model, scaler, _ = train_iris_model()
    test_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    test_scaled = scaler.transform(test_sample)
    prediction = model.predict(test_scaled)
    assert prediction[0] in [0, 1, 2]
