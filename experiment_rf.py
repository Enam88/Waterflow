# import mlflow
# import dvc.api
# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# import os

# # Set up an MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# # Initialize an MLflow experiment
# experiment_name = "experiment_water_quality_rf"
# experiment_id = mlflow.create_experiment(experiment_name)
# client = mlflow.tracking.MlflowClient()
# experiment = client.get_experiment(experiment_id)

# # Load data
# path = 'data/water_potability.csv'
# repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
# version = 'v1'

# data_url = dvc.api.get_url(path=path, repo=repo, rev=version)
# data = pd.read_csv(data_url)

# # Handle missing values
# numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
# imputer = SimpleImputer(strategy='mean')
# data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# # Split data into features and target
# X = data.drop('Potability', axis=1)
# y = data['Potability']

# # Split data into train and test sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Further split train data into train and validation sets (80% train, 20% validation)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Define preprocessing steps
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())
# ])

# # Define pipeline with preprocessing and model
# pipeline = Pipeline(steps=[
#     ('preprocessor', numeric_transformer),
#     ('classifier', RandomForestClassifier())
# ])

# # Perform cross-validation on the training set
# cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

# # Print cross-validation scores
# print("Cross-Validation Scores:", cv_scores)
# print("Mean CV Score:", cv_scores.mean())

# # Train the model on the full training set
# pipeline.fit(X_train, y_train)

# # Evaluate the model on the validation set
# y_pred_val = pipeline.predict(X_val)
# accuracy_val = accuracy_score(y_val, y_pred_val)
# precision_val = precision_score(y_val, y_pred_val)
# recall_val = recall_score(y_val, y_pred_val)

# print("Validation Set Metrics:")
# print("Accuracy:", accuracy_val)
# print("Precision:", precision_val)
# print("Recall:", recall_val)

# # Evaluate the model on the test set
# y_pred_test = pipeline.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_pred_test)
# precision_test = precision_score(y_test, y_pred_test)
# recall_test = recall_score(y_test, y_pred_test)

# print("Test Set Metrics:")
# print("Accuracy:", accuracy_test)
# print("Precision:", precision_test)
# print("Recall:", recall_test)

# # Log metrics and model in MLflow
# with mlflow.start_run(experiment_id=experiment_id):
#     mlflow.log_param('data_url', data_url)
#     mlflow.log_param('data_version', version)
#     mlflow.log_param('input_rows', data.shape[0])
#     mlflow.log_param('input_cols', data.shape[1])
    
#     mlflow.log_metric('accuracy_val', accuracy_val)
#     mlflow.log_metric('precision_val', precision_val)
#     mlflow.log_metric('recall_val', recall_val)

#     # Log artifacts: columns used for modeling
#     os.makedirs("artifacts", exist_ok=True)
#     X_train.columns.to_series().to_csv("artifacts/features.csv", header=False, index=False)
#     mlflow.log_artifact("artifacts/features.csv")

#     # Log the name of the target column
#     with open("artifacts/targets.csv", "w") as f:
#         f.write(y_train.name)
#     mlflow.log_artifact("artifacts/targets.csv")

# import mlflow
# import dvc.api
# import pandas as pd
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# import os

# # Set up an MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# # Initialize an MLflow experiment
# experiment_name = "experiment_water_quality_rf"
# experiment_id = mlflow.create_experiment(experiment_name)
# client = mlflow.tracking.MlflowClient()
# experiment = client.get_experiment(experiment_id)

# # Load data
# path = 'data/water_potability.csv'
# repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
# version = 'v1'

# data_url = dvc.api.get_url(path=path, repo=repo, rev=version)
# data = pd.read_csv(data_url)

# # Handle missing values
# numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
# imputer = SimpleImputer(strategy='mean')
# data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# # Split data into features and target
# X = data.drop('Potability', axis=1)
# y = data['Potability']

# # Split data into train and test sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Further split train data into train and validation sets (80% train, 20% validation)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Define preprocessing steps
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='mean')),
#     ('scaler', StandardScaler())
# ])

# # Define pipeline with preprocessing and model
# pipeline = Pipeline(steps=[
#     ('preprocessor', numeric_transformer),
#     ('classifier', RandomForestClassifier())
# ])

# # Define hyperparameter grid for RandomizedSearchCV
# param_grid = {
#     'classifier__n_estimators': [50, 100, 150],
#     'classifier__max_depth': [None, 10, 20],
#     'classifier__min_samples_split': [2, 5, 10],
#     'classifier__min_samples_leaf': [1, 2, 4],
#     'classifier__bootstrap': [True, False]
# }

# # Set up RandomizedSearchCV
# random_search = RandomizedSearchCV(pipeline,
#                                     param_distributions=param_grid, 
#                                     n_iter=10, cv=5, 
#                                     random_state=42, 
#                                     verbose=2,
#                                     n_jobs=-1)

# # Fit RandomizedSearchCV to the training data
# random_search.fit(X_train, y_train)

# # Get the best model from RandomizedSearchCV
# best_model = random_search.best_estimator_

# # Evaluate the best model on the validation set
# y_pred_val = best_model.predict(X_val)
# accuracy_val = accuracy_score(y_val, y_pred_val)
# precision_val = precision_score(y_val, y_pred_val)
# recall_val = recall_score(y_val, y_pred_val)

# print("Validation Set Metrics:")
# print("Accuracy:", accuracy_val)
# print("Precision:", precision_val)
# print("Recall:", recall_val)

# # Evaluate the best model on the test set
# y_pred_test = best_model.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_pred_test)
# precision_test = precision_score(y_test, y_pred_test)
# recall_test = recall_score(y_test, y_pred_test)

# print("Test Set Metrics:")
# print("Accuracy:", accuracy_test)
# print("Precision:", precision_test)
# print("Recall:", recall_test)

# # Log metrics and model in MLflow
# with mlflow.start_run(experiment_id=experiment_id):
#     mlflow.log_param('data_url', data_url)
#     mlflow.log_param('data_version', version)
#     mlflow.log_param('input_rows', data.shape[0])
#     mlflow.log_param('input_cols', data.shape[1])
#     mlflow.log_param('best_params', random_search.best_params_)
    
#     mlflow.log_metric('accuracy_val', accuracy_val)
#     mlflow.log_metric('precision_val', precision_val)
#     mlflow.log_metric('recall_val', recall_val)

#     # Log artifacts: columns used for modeling
#     os.makedirs("artifacts", exist_ok=True)
#     X_train.columns.to_series().to_csv("artifacts/features.csv", header=False, index=False)
#     mlflow.log_artifact("artifacts/features.csv")

#     # Log the name of the target column
#     with open("artifacts/targets.csv", "w") as f:
#         f.write(y_train.name)
#     mlflow.log_artifact("artifacts/targets.csv")


import mlflow
import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os

def set_mlflow_tracking():
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "experiment_water_quality_rf"
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
        ('classifier', RandomForestClassifier())
    ])
    return pipeline

def tune_hyperparameters(X_train, y_train, pipeline):
    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(pipeline,
                                       param_distributions=param_grid, 
                                       n_iter=10, cv=5, 
                                       random_state=42, 
                                       verbose=2,
                                       n_jobs=-1)
    random_search.fit(X_train, y_train)
    return random_search

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    return accuracy, precision, recall

def log_metrics(experiment_id, data_url, version, data_shape, best_params, val_metrics, test_metrics, feature_names, target_name):
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param('data_url', data_url)
        mlflow.log_param('data_version', version)
        mlflow.log_param('input_rows', data_shape[0])
        mlflow.log_param('input_cols', data_shape[1])
        mlflow.log_param('best_params', best_params)
        
        mlflow.log_metric('accuracy_val', val_metrics[0])
        mlflow.log_metric('precision_val', val_metrics[1])
        mlflow.log_metric('recall_val', val_metrics[2])
        
        mlflow.log_metric('accuracy_test', test_metrics[0])
        mlflow.log_metric('precision_test', test_metrics[1])
        mlflow.log_metric('recall_test', test_metrics[2])

        os.makedirs("artifacts", exist_ok=True)
        pd.Series(feature_names).to_csv("artifacts/features.csv", header=False, index=False)
        mlflow.log_artifact("artifacts/features.csv")

        with open("artifacts/targets.csv", "w") as f:
            f.write(target_name)
        mlflow.log_artifact("artifacts/targets.csv")

def main():
    experiment_id = set_mlflow_tracking()
    data_path = 'data/water_potability.csv'
    repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
    version = 'v1'

    data, data_url = load_data(data_path, repo, version)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data)
    pipeline = create_pipeline()
    random_search = tune_hyperparameters(X_train, y_train, pipeline)
    best_model = random_search.best_estimator_

    val_metrics = evaluate_model(best_model, X_val, y_val)
    test_metrics = evaluate_model(best_model, X_test, y_test)

    print("Validation Set Metrics:")
    print("Accuracy:", val_metrics[0])
    print("Precision:", val_metrics[1])
    print("Recall:", val_metrics[2])

    print("Test Set Metrics:")
    print("Accuracy:", test_metrics[0])
    print("Precision:", test_metrics[1])
    print("Recall:", test_metrics[2])

    log_metrics(experiment_id, data_url, version, data.shape, random_search.best_params_, val_metrics, test_metrics, X_train.columns.tolist(), y_train.name)

if __name__ == "__main__":
    main()
