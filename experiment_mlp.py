

import mlflow
import dvc.api
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import os
from sklearn.impute import SimpleImputer
from keras.wrappers.scikit_learn import KerasClassifier
from mlflow.models.signature import infer_signature
import joblib

# Custom wrapper for Keras model
class KerasWrapper(KerasClassifier):
    def __init__(self, build_fn=None, **sk_params):
        super().__init__(build_fn=build_fn, **sk_params)

    def predict(self, X, **kwargs):
        probabilities = self.model.predict(X, **kwargs)
        return (probabilities > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

def load_data(path, repo, version):
    data_url = dvc.api.get_url(path=path, repo=repo, rev=version)
    data = pd.read_csv(data_url)
    return data

def preprocess_data(data, training=True, scaler=None):
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='mean')
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    X = data.drop('Potability', axis=1)
    y = data['Potability'] if 'Potability' in data else None
    
    feature_names = X.columns.tolist()

    if training:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler
    else:
        X = scaler.transform(X)
        return X, feature_names

def create_model(input_shape, learning_rate=0.001, dropout_rate=0.5):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def tune_hyperparameters(X_train, y_train):
    model = KerasWrapper(build_fn=create_model, input_shape=X_train.shape[1])
    param_dist = {
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'epochs': [50, 100, 200]
    }
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        cv=5,
        n_iter=20,  # Increase the number of iterations
        scoring='accuracy',
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search.best_params_

def train_final_model(X_train, y_train, X_val, y_val, best_params):
    final_model = create_model(X_train.shape[1], learning_rate=best_params['learning_rate'], dropout_rate=best_params['dropout_rate'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    final_model.fit(X_train, y_train, epochs=best_params['epochs'], validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
    return final_model

def evaluate_and_log_model(model, X_test, y_test, feature_names, experiment_id, data_url, version, y_train, y_val, best_params, scaler):
    y_pred_test_proba = model.predict(X_test)
    y_pred_test = (y_pred_test_proba > 0.5).astype(int).flatten()
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, output_dict=True)
    confusion = confusion_matrix(y_test, y_pred_test)

    print("Test Set Metrics:")
    print("Accuracy:", accuracy_test)
    print("Precision:", precision_test)
    print("Recall:", recall_test)

    # Log model and metrics with MLflow
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param('data_url', data_url)
        mlflow.log_param('data_version', version)
        mlflow.log_param('input_rows', len(y_test) + len(y_train) + len(y_val))
        mlflow.log_param('input_cols', len(feature_names))
        mlflow.log_param('best_learning_rate', best_params['learning_rate'])
        mlflow.log_param('best_dropout_rate', best_params['dropout_rate'])
        mlflow.log_param('epochs', best_params['epochs'])
        mlflow.log_metric('accuracy_test', accuracy_test)
        mlflow.log_metric('precision_test', precision_test)
        mlflow.log_metric('recall_test', recall_test)

        os.makedirs("artifacts", exist_ok=True)
        pd.Series(feature_names).to_csv("artifacts/features.csv", header=False, index=False)
        mlflow.log_artifact("artifacts/features.csv")

        with open("artifacts/targets.csv", "w") as f:
            f.write(y_train.name)
        mlflow.log_artifact("artifacts/targets.csv")

        # Save the scaler
        joblib.dump(scaler, "artifacts/scaler.pkl")
        mlflow.log_artifact("artifacts/scaler.pkl")

        # Infer signature for the model
        signature = infer_signature(X_test, y_pred_test_proba)

        # Log the model with the signature
        mlflow.keras.log_model(model, "model", signature=signature)

        # Log classification report and confusion matrix
        mlflow.log_dict(report, "classification_report.json")
        mlflow.log_dict({'confusion_matrix': confusion.tolist()}, "confusion_matrix.json")

        # Register the model to the MLflow Model Registry
        model_uri = "runs:/{}/model".format(mlflow.active_run().info.run_id)
        mlflow.register_model(model_uri, "WaterQualityMLP")

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "experiment_water_quality_mlp"
    experiment_id = mlflow.create_experiment(experiment_name)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment(experiment_id)

    data_path = 'data/water_potability.csv'
    repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
    version = 'v1'

    data = load_data(data_path, repo, version)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler = preprocess_data(data)
    best_params = tune_hyperparameters(X_train, y_train)
    final_model = train_final_model(X_train, y_train, X_val, y_val, best_params)
    evaluate_and_log_model(final_model, X_test, y_test, feature_names, experiment_id, data_path, version, y_train, y_val, best_params, scaler)

if __name__ == "__main__":
    main()
