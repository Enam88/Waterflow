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

# Set up an MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Initialize an MLflow experiment
experiment_name = "experiment_water_quality_xgb"
experiment_id = mlflow.create_experiment(experiment_name)
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment(experiment_id)

# Load data
path = 'data/water_potability.csv'
repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
version = 'v1'

data_url = dvc.api.get_url(path=path, repo=repo, rev=version)
data = pd.read_csv(data_url)

# Handle missing values
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# Split data into features and target
X = data.drop('Potability', axis=1)
y = data['Potability']

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split train data into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', numeric_transformer),
    ('classifier', XGBClassifier())
])

# Define hyperparameter grid for RandomizedSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [3, 5, 10],
    'classifier__learning_rate': [0.01, 0.1, 0.3],
    'classifier__subsample': [0.5, 0.7, 1.0],
    'classifier__colsample_bytree': [0.5, 0.7, 1.0],
    'classifier__gamma': [0, 0.1, 0.3]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=5, random_state=42, verbose=2, n_jobs=-1)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# Get the best model from RandomizedSearchCV
best_model = random_search.best_estimator_

# Evaluate the best model on the validation set
y_pred_val = best_model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
precision_val = precision_score(y_val, y_pred_val)
recall_val = recall_score(y_val, y_pred_val)

print("Validation Set Metrics:")
print("Accuracy:", accuracy_val)
print("Precision:", precision_val)
print("Recall:", recall_val)

# Evaluate the best model on the test set
y_pred_test = best_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)

print("Test Set Metrics:")
print("Accuracy:", accuracy_test)
print("Precision:", precision_test)
print("Recall:", recall_test)

# Log metrics and model in MLflow
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_param('data_url', data_url)
    mlflow.log_param('data_version', version)
    mlflow.log_param('input_rows', data.shape[0])
    mlflow.log_param('input_cols', data.shape[1])
    mlflow.log_param('best_params', random_search.best_params_)
    
    mlflow.log_metric('accuracy_val', accuracy_val)
    mlflow.log_metric('precision_val', precision_val)
    mlflow.log_metric('recall_val', recall_val)

    # Log artifacts: columns used for modeling
    os.makedirs("artifacts", exist_ok=True)
    X_train.columns.to_series().to_csv("artifacts/features.csv", header=False, index=False)
    mlflow.log_artifact("artifacts/features.csv")

    # Log the name of the target column
    with open("artifacts/targets.csv", "w") as f:
        f.write(y_train.name)
    mlflow.log_artifact("artifacts/targets.csv")
