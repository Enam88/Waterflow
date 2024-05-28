

import mlflow
import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import os

def set_mlflow_tracking():
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "experiment_water_quality_xgb"
    experiment_id = mlflow.create_experiment(experiment_name)
    return experiment_id

def load_data(path, repo, version):
    data_url = dvc.api.get_url(path=path, repo=repo, rev=version)
    data = pd.read_csv(data_url)
    return data, data_url

def preprocess_data(data):
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='mean')
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    X = data.drop('Potability', axis=1)
    y = data['Potability']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_pipeline():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', numeric_transformer),
        ('classifier', XGBClassifier())
    ])

    return pipeline

def tune_hyperparameters(X_train, y_train, pipeline):
    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [3, 5, 10],
        'classifier__learning_rate': [0.01, 0.1, 0.3],
        'classifier__subsample': [0.5, 0.7, 1.0],
        'classifier__colsample_bytree': [0.5, 0.7, 1.0],
        'classifier__gamma': [0, 0.1, 0.3]
    }

    random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=5, random_state=42, verbose=2, n_jobs=-1)
    random_search.fit(X_train, y_train)
    
    return random_search

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    return accuracy, precision, recall

def log_metrics_and_params(experiment_id, data_url, version, data_shape, best_params, accuracy_val, precision_val, recall_val):
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param('data_url', data_url)
        mlflow.log_param('data_version', version)
        mlflow.log_param('input_rows', data_shape[0])
        mlflow.log_param('input_cols', data_shape[1])
        mlflow.log_param('best_params', best_params)
        
        mlflow.log_metric('accuracy_val', accuracy_val)
        mlflow.log_metric('precision_val', precision_val)
        mlflow.log_metric('recall_val', recall_val)

        os.makedirs("artifacts", exist_ok=True)
        mlflow.log_artifact("artifacts/features.csv")

        with open("artifacts/targets.csv", "w") as f:
            f.write('Potability')
        mlflow.log_artifact("artifacts/targets.csv")

def main():
    experiment_id = set_mlflow_tracking()
    path = 'data/water_potability.csv'
    repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
    version = 'v1'

    data, data_url = load_data(path, repo, version)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data)
    pipeline = create_pipeline()
    random_search = tune_hyperparameters(X_train, y_train, pipeline)
    best_model = random_search.best_estimator_

    accuracy_val, precision_val, recall_val = evaluate_model(best_model, X_val, y_val)
    print("Validation Set Metrics:")
    print("Accuracy:", accuracy_val)
    print("Precision:", precision_val)
    print("Recall:", recall_val)

    accuracy_test, precision_test, recall_test = evaluate_model(best_model, X_test, y_test)
    print("Test Set Metrics:")
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)

    log_metrics_and_params(experiment_id, data_url, version, data.shape, random_search.best_params_, accuracy_val, precision_val, recall_val)

if __name__ == "__main__":
    main()
