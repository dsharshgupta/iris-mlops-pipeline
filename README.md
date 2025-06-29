# IRIS MLOps Pipeline

This project demonstrates MLOps best practices using the IRIS dataset.

## Features
- Data validation with pytest
- Model training and evaluation  
- DVC for data versioning
- GitHub Actions for CI/CD
- CML for model performance reporting

## Run Pipeline
```bash
python src/data_loader.py
python src/train_model.py
python src/evaluate_model.py
