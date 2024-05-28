import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from experiment_rf import load_data, preprocess_data, create_pipeline, tune_hyperparameters, evaluate_model, set_mlflow_tracking

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [2, 3, 4, 5, 6],
        'Potability': [0, 1, 0, 1, 0]
    })
    return data

def test_set_mlflow_tracking():
    experiment_id = set_mlflow_tracking()
    assert isinstance(experiment_id, str)
    assert len(experiment_id) > 0

def test_load_data():
    path = 'data/water_potability.csv'
    repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
    version = 'v1'
    data, data_url = load_data(path, repo, version)
    assert not data.empty
    assert isinstance(data_url, str)
    assert len(data_url) > 0

def test_preprocess_data(sample_data):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(sample_data)
    assert X_train.shape[1] == 2
    assert X_val.shape[1] == 2
    assert X_test.shape[1] == 2
    assert len(y_train) > 0
    assert len(y_val) > 0
    assert len(y_test) > 0

def test_create_pipeline():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert 'classifier' in pipeline.named_steps

def test_tune_hyperparameters(sample_data):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(sample_data)
    pipeline = create_pipeline()
    random_search = tune_hyperparameters(X_train, y_train, pipeline)
    assert 'classifier__n_estimators' in random_search.best_params_

def test_evaluate_model(sample_data):
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(sample_data)
    pipeline = create_pipeline()
    random_search = tune_hyperparameters(X_train, y_train, pipeline)
    best_model = random_search.best_estimator_
    accuracy_val, precision_val, recall_val = evaluate_model(best_model, X_val, y_val)
    assert accuracy_val > 0
    assert precision_val > 0
    assert recall_val > 0

    accuracy_test, precision_test, recall_test = evaluate_model(best_model, X_test, y_test)
    assert accuracy_test > 0
    assert precision_test > 0
    assert recall_test > 0

def test_main():
    from experiment_rf import main
    main()
