import pandas as pd
from sklearn.datasets import load_iris
import os

def load_iris_data():
    """Load the IRIS dataset and save to CSV"""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/iris.csv', index=False)
    print(f"Data saved to data/iris.csv with shape: {df.shape}")
    return df

def validate_data(df):
    """Validate the loaded data"""
    assert df.isnull().sum().sum() == 0, "Data contains missing values"
    assert df.shape[0] == 150, f"Expected 150 rows, got {df.shape[0]}"
    assert df.shape[1] == 6, f"Expected 6 columns, got {df.shape[1]}"
    
    species_counts = df['species'].value_counts()
    assert all(species_counts == 50), "Each species should have 50 samples"
    
    print("Data validation passed!")
    return True

if __name__ == "__main__":
    df = load_iris_data()
    validate_data(df)
