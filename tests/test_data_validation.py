import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import load_iris_data, validate_data

def test_data_shape():
    """Test if data has correct shape"""
    df = load_iris_data()
    assert df.shape[0] == 150
    assert df.shape[1] == 6

def test_no_missing_values():
    """Test for missing values"""
    df = load_iris_data()
    assert df.isnull().sum().sum() == 0

def test_species_distribution():
    """Test species distribution"""
    df = load_iris_data()
    species_counts = df['species'].value_counts()
    assert all(species_counts == 50)

def test_validate_data_function():
    """Test validate_data function"""
    df = load_iris_data()
    assert validate_data(df) == True
