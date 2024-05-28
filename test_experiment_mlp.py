import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from experiment_mlp import (
    load_data, preprocess_data, create_model, tune_hyperparameters, 
    train_final_model, evaluate_and_log_model, KerasWrapper
)

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [2, 3, 4, 5, 6],
        'Potability': [0, 1, 0, 1, 0]
    })
    return data

def test_load_data(mocker):
    mocker.patch('dvc.api.get_url', return_value='dummy_url')
    mocker.patch('pandas.read_csv', return_value=pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Potability': np.random.randint(0, 2, 100)
    }))
    path = 'data/water_potability.csv'
    repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
    version = 'v1'
    data = load_data(path, repo, version)
    assert not data.empty

def test_preprocess_data(sample_data):
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_data(sample_data)
    assert X_train.shape[1] == 2
    assert X_val.shape[1] == 2
    assert X_test.shape[1] == 2
    assert len(y_train) > 0
    assert len(y_val) > 0
    assert len(y_test) > 0
    assert feature_names == ['Feature1', 'Feature2']

def test_create_model():
    model = create_model(input_shape=2)
    assert model.count_params() > 0

def test_tune_hyperparameters(sample_data):
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_data(sample_data)
    best_params = tune_hyperparameters(X_train, y_train)
    assert 'learning_rate' in best_params
    assert 'dropout_rate' in best_params

def test_train_final_model(sample_data):
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_data(sample_data)
    best_params = tune_hyperparameters(X_train, y_train)
    model = train_final_model(X_train, y_train, X_val, y_val, best_params)
    assert model.count_params() > 0

def test_evaluate_and_log_model(mocker, sample_data):
    # Mocking mlflow functions to avoid actual MLflow server calls during testing
    mocker.patch('experiment_mlp.mlflow.start_run')
    mocker.patch('experiment_mlp.mlflow.log_param')
    mocker.patch('experiment_mlp.mlflow.log_metric')
    mocker.patch('experiment_mlp.mlflow.log_artifact')
    mocker.patch('os.makedirs')

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_data(sample_data)
    best_params = tune_hyperparameters(X_train, y_train)
    model = train_final_model(X_train, y_train, X_val, y_val, best_params)

    # Call evaluate_and_log_model
    evaluate_and_log_model(
        model, X_test, y_test, feature_names,
        experiment_id=1, data_url='dummy_url', version='v1',
        y_train=y_train, y_val=y_val, best_params=best_params
    )

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    assert accuracy > 0
    assert precision > 0
    assert recall > 0

def test_main(mocker):
    # Mocking functions to avoid actual MLflow server calls and data loading during testing
    mocker.patch('experiment_mlp.mlflow.set_tracking_uri')
    mocker.patch('experiment_mlp.mlflow.create_experiment', return_value=1)
    mocker.patch('experiment_mlp.mlflow.tracking.MlflowClient')
    mocker.patch('experiment_mlp.load_data', return_value=pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Potability': np.random.randint(0, 2, 100)
    }))
    mocker.patch('experiment_mlp.evaluate_and_log_model')
    mocker.patch('os.makedirs')

    from experiment_mlp import main
    main()

if __name__ == "__main__":
    pytest.main()
