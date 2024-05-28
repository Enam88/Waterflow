# import mlflow
# import dvc.api
# import pandas as pd
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from tensorflow import keras
# from keras.wrappers.scikit_learn import KerasClassifier
# import os

# # Set up an MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# # Initialize an MLflow experiment
# experiment_name = "experiment_water_quality_mlp"
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

# # Define function to create MLP model
# def create_mlp_model():
#     model = keras.Sequential([
#         keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#         keras.layers.Dense(32, activation='relu'),
#         keras.layers.Dense(1, activation='sigmoid')
#     ])

#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# # Define pipeline with preprocessing and model
# pipeline = Pipeline(steps=[
#     ('preprocessor', numeric_transformer),
#     ('classifier', KerasClassifier(build_fn=create_mlp_model, verbose=0))
# ])

# # Define hyperparameter grid for RandomizedSearchCV
# param_grid = {
#     'classifier__batch_size': [16, 32, 64],
#     'classifier__epochs': [10, 20, 30]
# }

# # Set up RandomizedSearchCV
# random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=5, random_state=42, verbose=2, n_jobs=-1)

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

# import mlflow
# import dvc.api
# import pandas as pd
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from tensorflow import keras
# from keras.wrappers.scikit_learn import KerasClassifier
# import os

# # Set up an MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# # Initialize an MLflow experiment
# experiment_name = "experiment_water_quality_mlp"
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

# # Define function to create MLP model
# def create_mlp_model(dropout_rate=0.2):
#     model = keras.Sequential([
#         keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#         keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(32, activation='relu'),
#         keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(16, activation='relu'),
#         keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(1, activation='sigmoid')
#     ])

#     model.compile(optimizer='adam',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model

# # Define pipeline with preprocessing and model
# pipeline = Pipeline(steps=[
#     ('preprocessor', numeric_transformer),
#     ('classifier', KerasClassifier(build_fn=create_mlp_model, verbose=0))
# ])

# # Define hyperparameter grid for RandomizedSearchCV
# param_grid = {
#     'classifier__batch_size': [16, 32, 64],
#     'classifier__epochs': [10, 20, 30],
#     'classifier__dropout_rate': [0.1, 0.2, 0.3]
# }

# # Set up RandomizedSearchCV
# random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=10, cv=5, random_state=42, verbose=2, n_jobs=-1)

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



# import mlflow
# import dvc.api
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
# import os
# from sklearn.impute import SimpleImputer


# # Set up an MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# # Initialize an MLflow experiment
# experiment_name = "experiment_water_quality_mlp"
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

# # Define the MLP model architecture with dropout and batch normalization
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.5),  # Dropout regularization
#     BatchNormalization(),  # Batch normalization
#     Dense(32, activation='relu'),
#     Dropout(0.5),  # Dropout regularization
#     BatchNormalization(),  # Batch normalization
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model with Adam optimizer and binary cross-entropy loss
# model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# # Define early stopping callback to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train the model with early stopping
# history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# # Evaluate the model on the validation set
# y_pred_val = (model.predict(X_val) > 0.5).astype(int)
# accuracy_val = accuracy_score(y_val, y_pred_val)
# precision_val = precision_score(y_val, y_pred_val)
# recall_val = recall_score(y_val, y_pred_val)

# print("Validation Set Metrics:")
# print("Accuracy:", accuracy_val)
# print("Precision:", precision_val)
# print("Recall:", recall_val)

# # Evaluate the model on the test set
# y_pred_test = (model.predict(X_test) > 0.5).astype(int)
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
#     mlflow.log_metric('accuracy_test', accuracy_test)
#     mlflow.log_metric('precision_test', precision_test)
#     mlflow.log_metric('recall_test', recall_test)

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
# import numpy as np
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam
# import os
# from sklearn.impute import SimpleImputer
# from keras.wrappers.scikit_learn import KerasClassifier

# # Custom wrapper for Keras model
# class KerasWrapper(KerasClassifier):
#     def __init__(self, build_fn=None, **sk_params):
#         super().__init__(build_fn=build_fn, **sk_params)

#     def predict(self, X, **kwargs):
#         probabilities = self.model.predict(X, **kwargs)
#         return (probabilities > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

# def load_data(path, repo, version):
#     data_url = dvc.api.get_url(path=path, repo=repo, rev=version)
#     data = pd.read_csv(data_url)
#     return data

# def preprocess_data(data):
#     numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
#     imputer = SimpleImputer(strategy='mean')
#     data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

#     X = data.drop('Potability', axis=1)
#     y = data['Potability']
    
#     feature_names = X.columns.tolist()
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)
#     X_test = scaler.transform(X_test)

#     return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

# def create_model(input_shape, learning_rate=0.001, dropout_rate=0.5):
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(input_shape,)),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(1, activation='sigmoid')
#     ])
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def tune_hyperparameters(X_train, y_train):
#     model = KerasWrapper(build_fn=create_model, input_shape=X_train.shape[1])
#     param_dist = {
#         'learning_rate': [0.0001, 0.001, 0.01],
#         'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
#     }
#     random_search = RandomizedSearchCV(
#         estimator=model,
#         param_distributions=param_dist,
#         cv=5,
#         n_iter=10,
#         scoring='accuracy',
#         n_jobs=-1
#     )
#     random_search.fit(X_train, y_train, epochs=100, verbose=0)
#     return random_search.best_params_

# def train_final_model(X_train, y_train, X_val, y_val, best_params):
#     final_model = create_model(X_train.shape[1], learning_rate=best_params['learning_rate'], dropout_rate=best_params['dropout_rate'])
#     final_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0)
#     return final_model

# def evaluate_and_log_model(model, X_test, y_test, feature_names, experiment_id, data_url, version, y_train, y_val, best_params):
#     y_pred_test_proba = model.predict(X_test)
#     y_pred_test = (y_pred_test_proba > 0.5).astype(int).flatten()
#     accuracy_test = accuracy_score(y_test, y_pred_test)
#     precision_test = precision_score(y_test, y_pred_test)
#     recall_test = recall_score(y_test, y_pred_test)

#     print("Test Set Metrics:")
#     print("Accuracy:", accuracy_test)
#     print("Precision:", precision_test)
#     print("Recall:", recall_test)

#     with mlflow.start_run(experiment_id=experiment_id):
#         mlflow.log_param('data_url', data_url)
#         mlflow.log_param('data_version', version)
#         mlflow.log_param('input_rows', len(y_test) + len(y_train) + len(y_val))
#         mlflow.log_param('input_cols', len(feature_names))
#         mlflow.log_param('best_learning_rate', best_params['learning_rate'])
#         mlflow.log_param('best_dropout_rate', best_params['dropout_rate'])
#         mlflow.log_metric('accuracy_test', accuracy_test)
#         mlflow.log_metric('precision_test', precision_test)
#         mlflow.log_metric('recall_test', recall_test)

#         os.makedirs("artifacts", exist_ok=True)
#         pd.Series(feature_names).to_csv("artifacts/features.csv", header=False, index=False)
#         mlflow.log_artifact("artifacts/features.csv")

#         with open("artifacts/targets.csv", "w") as f:
#             f.write(y_train.name)
#         mlflow.log_artifact("artifacts/targets.csv")

# def main():
#     mlflow.set_tracking_uri("http://localhost:5000")
#     experiment_name = "experiment_water_quality_mlp"
#     experiment_id = mlflow.create_experiment(experiment_name)
#     client = mlflow.tracking.MlflowClient()
#     experiment = client.get_experiment(experiment_id)

#     data_path = 'data/water_potability.csv'
#     repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
#     version = 'v1'

#     data = load_data(data_path, repo, version)
#     X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_data(data)
#     best_params = tune_hyperparameters(X_train, y_train)
#     final_model = train_final_model(X_train, y_train, X_val, y_val, best_params)
#     evaluate_and_log_model(final_model, X_test, y_test, feature_names, experiment_id, data_path, version, y_train, y_val, best_params)

# if __name__ == "__main__":
#     main()


# import mlflow
# import dvc.api
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam
# import os
# from sklearn.impute import SimpleImputer
# from keras.wrappers.scikit_learn import KerasClassifier
# from mlflow.models.signature import infer_signature

# # Custom wrapper for Keras model
# class KerasWrapper(KerasClassifier):
#     def __init__(self, build_fn=None, **sk_params):
#         super().__init__(build_fn=build_fn, **sk_params)

#     def predict(self, X, **kwargs):
#         probabilities = self.model.predict(X, **kwargs)
#         return (probabilities > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

# def load_data(path, repo, version):
#     data_url = dvc.api.get_url(path=path, repo=repo, rev=version)
#     data = pd.read_csv(data_url)
#     return data

# def preprocess_data(data):
#     numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
#     imputer = SimpleImputer(strategy='mean')
#     data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

#     X = data.drop('Potability', axis=1)
#     y = data['Potability']
    
#     feature_names = X.columns.tolist()
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)
#     X_test = scaler.transform(X_test)

#     return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

# def create_model(input_shape, learning_rate=0.001, dropout_rate=0.5):
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(input_shape,)),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(1, activation='sigmoid')
#     ])
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def tune_hyperparameters(X_train, y_train):
#     model = KerasWrapper(build_fn=create_model, input_shape=X_train.shape[1])
#     param_dist = {
#         'learning_rate': [0.0001, 0.001, 0.01],
#         'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
#     }
#     random_search = RandomizedSearchCV(
#         estimator=model,
#         param_distributions=param_dist,
#         cv=5,
#         n_iter=10,
#         scoring='accuracy',
#         n_jobs=-1
#     )
#     random_search.fit(X_train, y_train, epochs=100, verbose=0)
#     return random_search.best_params_

# def train_final_model(X_train, y_train, X_val, y_val, best_params):
#     final_model = create_model(X_train.shape[1], learning_rate=best_params['learning_rate'], dropout_rate=best_params['dropout_rate'])
#     final_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0)
#     return final_model

# def evaluate_and_log_model(model, X_test, y_test, feature_names, experiment_id, data_url, version, y_train, y_val, best_params):
#     y_pred_test_proba = model.predict(X_test)
#     y_pred_test = (y_pred_test_proba > 0.5).astype(int).flatten()
#     accuracy_test = accuracy_score(y_test, y_pred_test)
#     precision_test = precision_score(y_test, y_pred_test)
#     recall_test = recall_score(y_test, y_pred_test)

#     print("Test Set Metrics:")
#     print("Accuracy:", accuracy_test)
#     print("Precision:", precision_test)
#     print("Recall:", recall_test)

#     # Log model and metrics with MLflow
#     with mlflow.start_run(experiment_id=experiment_id):
#         mlflow.log_param('data_url', data_url)
#         mlflow.log_param('data_version', version)
#         mlflow.log_param('input_rows', len(y_test) + len(y_train) + len(y_val))
#         mlflow.log_param('input_cols', len(feature_names))
#         mlflow.log_param('best_learning_rate', best_params['learning_rate'])
#         mlflow.log_param('best_dropout_rate', best_params['dropout_rate'])
#         mlflow.log_metric('accuracy_test', accuracy_test)
#         mlflow.log_metric('precision_test', precision_test)
#         mlflow.log_metric('recall_test', recall_test)

#         os.makedirs("artifacts", exist_ok=True)
#         pd.Series(feature_names).to_csv("artifacts/features.csv", header=False, index=False)
#         mlflow.log_artifact("artifacts/features.csv")

#         with open("artifacts/targets.csv", "w") as f:
#             f.write(y_train.name)
#         mlflow.log_artifact("artifacts/targets.csv")

#         # Infer signature for the model
#         signature = infer_signature(X_test, y_pred_test_proba)

#         # Log the model with the signature
#         mlflow.keras.log_model(model.model, "model", signature=signature)

# def main():
#     mlflow.set_tracking_uri("http://localhost:5000")
#     experiment_name = "experiment_water_quality_mlp"
#     experiment_id = mlflow.create_experiment(experiment_name)
#     client = mlflow.tracking.MlflowClient()
#     experiment = client.get_experiment(experiment_id)

#     data_path = 'data/water_potability.csv'
#     repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
#     version = 'v1'

#     data = load_data(data_path, repo, version)
#     X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_data(data)
#     best_params = tune_hyperparameters(X_train, y_train)
#     final_model = train_final_model(X_train, y_train, X_val, y_val, best_params)
#     evaluate_and_log_model(final_model, X_test, y_test, feature_names, experiment_id, data_path, version, y_train, y_val, best_params)

# if __name__ == "__main__":
#     main()


# import mlflow
# import dvc.api
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam
# import os
# from sklearn.impute import SimpleImputer
# from keras.wrappers.scikit_learn import KerasClassifier
# from mlflow.models.signature import infer_signature

# # Custom wrapper for Keras model
# class KerasWrapper(KerasClassifier):
#     def __init__(self, build_fn=None, **sk_params):
#         super().__init__(build_fn=build_fn, **sk_params)

#     def predict(self, X, **kwargs):
#         probabilities = self.model.predict(X, **kwargs)
#         return (probabilities > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

# def load_data(path, repo, version):
#     data_url = dvc.api.get_url(path=path, repo=repo, rev=version)
#     data = pd.read_csv(data_url)
#     return data

# def preprocess_data(data):
#     numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
#     imputer = SimpleImputer(strategy='mean')
#     data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

#     X = data.drop('Potability', axis=1)
#     y = data['Potability']
    
#     feature_names = X.columns.tolist()
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)
#     X_test = scaler.transform(X_test)

#     return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

# def create_model(input_shape, learning_rate=0.001, dropout_rate=0.5):
#     model = Sequential([
#         Dense(64, activation='relu', input_shape=(input_shape,)),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(1, activation='sigmoid')
#     ])
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def tune_hyperparameters(X_train, y_train):
#     model = KerasWrapper(build_fn=create_model, input_shape=X_train.shape[1])
#     param_dist = {
#         'learning_rate': [0.0001, 0.001, 0.01],
#         'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5]
#     }
#     random_search = RandomizedSearchCV(
#         estimator=model,
#         param_distributions=param_dist,
#         cv=5,
#         n_iter=10,
#         scoring='accuracy',
#         n_jobs=-1
#     )
#     random_search.fit(X_train, y_train, epochs=100, verbose=0)
#     return random_search.best_params_

# def train_final_model(X_train, y_train, X_val, y_val, best_params):
#     final_model = create_model(X_train.shape[1], learning_rate=best_params['learning_rate'], dropout_rate=best_params['dropout_rate'])
#     final_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=0)
#     return final_model

# def evaluate_and_log_model(model, X_test, y_test, feature_names, experiment_id, data_url, version, y_train, y_val, best_params):
#     y_pred_test_proba = model.predict(X_test)
#     y_pred_test = (y_pred_test_proba > 0.5).astype(int).flatten()
#     accuracy_test = accuracy_score(y_test, y_pred_test)
#     precision_test = precision_score(y_test, y_pred_test)
#     recall_test = recall_score(y_test, y_pred_test)

#     print("Test Set Metrics:")
#     print("Accuracy:", accuracy_test)
#     print("Precision:", precision_test)
#     print("Recall:", recall_test)

#     # Log model and metrics with MLflow
#     with mlflow.start_run(experiment_id=experiment_id):
#         mlflow.log_param('data_url', data_url)
#         mlflow.log_param('data_version', version)
#         mlflow.log_param('input_rows', len(y_test) + len(y_train) + len(y_val))
#         mlflow.log_param('input_cols', len(feature_names))
#         mlflow.log_param('best_learning_rate', best_params['learning_rate'])
#         mlflow.log_param('best_dropout_rate', best_params['dropout_rate'])
#         mlflow.log_metric('accuracy_test', accuracy_test)
#         mlflow.log_metric('precision_test', precision_test)
#         mlflow.log_metric('recall_test', recall_test)

#         os.makedirs("artifacts", exist_ok=True)
#         pd.Series(feature_names).to_csv("artifacts/features.csv", header=False, index=False)
#         mlflow.log_artifact("artifacts/features.csv")

#         with open("artifacts/targets.csv", "w") as f:
#             f.write(y_train.name)
#         mlflow.log_artifact("artifacts/targets.csv")

#         # Infer signature for the model
#         signature = infer_signature(X_test, y_pred_test_proba)

#         # Log the model with the signature
#         mlflow.keras.log_model(model, "model", signature=signature)

# def main():
#     mlflow.set_tracking_uri("http://localhost:5000")
#     experiment_name = "experiment_water_quality_mlp"
#     experiment_id = mlflow.create_experiment(experiment_name)
#     client = mlflow.tracking.MlflowClient()
#     experiment = client.get_experiment(experiment_id)

#     data_path = 'data/water_potability.csv'
#     repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
#     version = 'v1'

#     data = load_data(data_path, repo, version)
#     X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_data(data)
#     best_params = tune_hyperparameters(X_train, y_train)
#     final_model = train_final_model(X_train, y_train, X_val, y_val, best_params)
#     evaluate_and_log_model(final_model, X_test, y_test, feature_names, experiment_id, data_path, version, y_train, y_val, best_params)

# if __name__ == "__main__":
#     main()



# import mlflow
# import dvc.api
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
# from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam
# import os
# from sklearn.impute import SimpleImputer
# from keras.wrappers.scikit_learn import KerasClassifier
# from mlflow.models.signature import infer_signature

# # Custom wrapper for Keras model
# class KerasWrapper(KerasClassifier):
#     def __init__(self, build_fn=None, **sk_params):
#         super().__init__(build_fn=build_fn, **sk_params)

#     def predict(self, X, **kwargs):
#         probabilities = self.model.predict(X, **kwargs)
#         return (probabilities > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

# def load_data(path, repo, version):
#     data_url = dvc.api.get_url(path=path, repo=repo, rev=version)
#     data = pd.read_csv(data_url)
#     return data

# def preprocess_data(data):
#     numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
#     imputer = SimpleImputer(strategy='mean')
#     data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

#     X = data.drop('Potability', axis=1)
#     y = data['Potability']
    
#     feature_names = X.columns.tolist()
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)
#     X_test = scaler.transform(X_test)

#     return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

# def create_model(input_shape, learning_rate=0.001, dropout_rate=0.5):
#     model = Sequential([
#         Dense(128, activation='relu', input_shape=(input_shape,)),
#         Dropout(dropout_rate),
#         Dense(64, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(1, activation='sigmoid')
#     ])
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def tune_hyperparameters(X_train, y_train):
#     model = KerasWrapper(build_fn=create_model, input_shape=X_train.shape[1])
#     param_dist = {
#         'learning_rate': [0.0001, 0.001, 0.01, 0.1],
#         'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
#         'epochs': [50, 100, 200]
#     }
#     random_search = RandomizedSearchCV(
#         estimator=model,
#         param_distributions=param_dist,
#         cv=5,
#         n_iter=20,  # Increase the number of iterations
#         scoring='accuracy',
#         n_jobs=-1
#     )
#     random_search.fit(X_train, y_train)
#     return random_search.best_params_

# def train_final_model(X_train, y_train, X_val, y_val, best_params):
#     final_model = create_model(X_train.shape[1], learning_rate=best_params['learning_rate'], dropout_rate=best_params['dropout_rate'])
#     final_model.fit(X_train, y_train, epochs=best_params['epochs'], validation_data=(X_val, y_val), verbose=0)
#     return final_model

# def evaluate_and_log_model(model, X_test, y_test, feature_names, experiment_id, data_url, version, y_train, y_val, best_params):
#     y_pred_test_proba = model.predict(X_test)
#     y_pred_test = (y_pred_test_proba > 0.5).astype(int).flatten()
#     accuracy_test = accuracy_score(y_test, y_pred_test)
#     precision_test = precision_score(y_test, y_pred_test)
#     recall_test = recall_score(y_test, y_pred_test)
#     report = classification_report(y_test, y_pred_test, output_dict=True)
#     confusion = confusion_matrix(y_test, y_pred_test)

#     print("Test Set Metrics:")
#     print("Accuracy:", accuracy_test)
#     print("Precision:", precision_test)
#     print("Recall:", recall_test)

#     # Log model and metrics with MLflow
#     with mlflow.start_run(experiment_id=experiment_id):
#         mlflow.log_param('data_url', data_url)
#         mlflow.log_param('data_version', version)
#         mlflow.log_param('input_rows', len(y_test) + len(y_train) + len(y_val))
#         mlflow.log_param('input_cols', len(feature_names))
#         mlflow.log_param('best_learning_rate', best_params['learning_rate'])
#         mlflow.log_param('best_dropout_rate', best_params['dropout_rate'])
#         mlflow.log_param('epochs', best_params['epochs'])
#         mlflow.log_metric('accuracy_test', accuracy_test)
#         mlflow.log_metric('precision_test', precision_test)
#         mlflow.log_metric('recall_test', recall_test)

#         os.makedirs("artifacts", exist_ok=True)
#         pd.Series(feature_names).to_csv("artifacts/features.csv", header=False, index=False)
#         mlflow.log_artifact("artifacts/features.csv")

#         with open("artifacts/targets.csv", "w") as f:
#             f.write(y_train.name)
#         mlflow.log_artifact("artifacts/targets.csv")

#         # Infer signature for the model
#         signature = infer_signature(X_test, y_pred_test_proba)

#         # Log the model with the signature
#         mlflow.keras.log_model(model, "model", signature=signature)

#         # Log classification report and confusion matrix
#         mlflow.log_dict(report, "classification_report.json")
#         mlflow.log_dict({'confusion_matrix': confusion.tolist()}, "confusion_matrix.json")

# def main():
#     mlflow.set_tracking_uri("http://localhost:5000")
#     experiment_name = "experiment_water_quality_mlp"
#     experiment_id = mlflow.create_experiment(experiment_name)
#     client = mlflow.tracking.MlflowClient()
#     experiment = client.get_experiment(experiment_id)

#     data_path = 'data/water_potability.csv'
#     repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
#     version = 'v1'

#     data = load_data(data_path, repo, version)
#     X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_data(data)
#     best_params = tune_hyperparameters(X_train, y_train)
#     final_model = train_final_model(X_train, y_train, X_val, y_val, best_params)
#     evaluate_and_log_model(final_model, X_test, y_test, feature_names, experiment_id, data_path, version, y_train, y_val, best_params)

# if __name__ == "__main__":
#     main()

# import mlflow
# import dvc.api
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
# from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
# import os
# from sklearn.impute import SimpleImputer
# from keras.wrappers.scikit_learn import KerasClassifier
# from mlflow.models.signature import infer_signature

# # Custom wrapper for Keras model
# class KerasWrapper(KerasClassifier):
#     def __init__(self, build_fn=None, **sk_params):
#         super().__init__(build_fn=build_fn, **sk_params)

#     def predict(self, X, **kwargs):
#         probabilities = self.model.predict(X, **kwargs)
#         return (probabilities > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

# def load_data(path, repo, version):
#     data_url = dvc.api.get_url(path=path, repo=repo, rev=version)
#     data = pd.read_csv(data_url)
#     return data

# def preprocess_data(data):
#     numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
#     imputer = SimpleImputer(strategy='mean')
#     data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

#     X = data.drop('Potability', axis=1)
#     y = data['Potability']
    
#     feature_names = X.columns.tolist()
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)
#     X_test = scaler.transform(X_test)

#     return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

# def create_model(input_shape, learning_rate=0.001, dropout_rate=0.5):
#     model = Sequential([
#         Dense(128, activation='relu', input_shape=(input_shape,)),
#         Dropout(dropout_rate),
#         Dense(64, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(1, activation='sigmoid')
#     ])
#     optimizer = Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def tune_hyperparameters(X_train, y_train):
#     model = KerasWrapper(build_fn=create_model, input_shape=X_train.shape[1])
#     param_dist = {
#         'learning_rate': [0.0001, 0.001, 0.01, 0.1],
#         'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
#         'epochs': [50, 100, 200]
#     }
#     random_search = RandomizedSearchCV(
#         estimator=model,
#         param_distributions=param_dist,
#         cv=5,
#         n_iter=20,  # Increase the number of iterations
#         scoring='accuracy',
#         n_jobs=-1
#     )
#     random_search.fit(X_train, y_train)
#     return random_search.best_params_

# def train_final_model(X_train, y_train, X_val, y_val, best_params):
#     final_model = create_model(X_train.shape[1], learning_rate=best_params['learning_rate'], dropout_rate=best_params['dropout_rate'])
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     final_model.fit(X_train, y_train, epochs=best_params['epochs'], validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
#     return final_model

# def evaluate_and_log_model(model, X_test, y_test, feature_names, experiment_id, data_url, version, y_train, y_val, best_params):
#     y_pred_test_proba = model.predict(X_test)
#     y_pred_test = (y_pred_test_proba > 0.5).astype(int).flatten()
#     accuracy_test = accuracy_score(y_test, y_pred_test)
#     precision_test = precision_score(y_test, y_pred_test)
#     recall_test = recall_score(y_test, y_pred_test)
#     report = classification_report(y_test, y_pred_test, output_dict=True)
#     confusion = confusion_matrix(y_test, y_pred_test)

#     print("Test Set Metrics:")
#     print("Accuracy:", accuracy_test)
#     print("Precision:", precision_test)
#     print("Recall:", recall_test)

#     # Log model and metrics with MLflow
#     with mlflow.start_run(experiment_id=experiment_id):
#         mlflow.log_param('data_url', data_url)
#         mlflow.log_param('data_version', version)
#         mlflow.log_param('input_rows', len(y_test) + len(y_train) + len(y_val))
#         mlflow.log_param('input_cols', len(feature_names))
#         mlflow.log_param('best_learning_rate', best_params['learning_rate'])
#         mlflow.log_param('best_dropout_rate', best_params['dropout_rate'])
#         mlflow.log_param('epochs', best_params['epochs'])
#         mlflow.log_metric('accuracy_test', accuracy_test)
#         mlflow.log_metric('precision_test', precision_test)
#         mlflow.log_metric('recall_test', recall_test)

#         os.makedirs("artifacts", exist_ok=True)
#         pd.Series(feature_names).to_csv("artifacts/features.csv", header=False, index=False)
#         mlflow.log_artifact("artifacts/features.csv")

#         with open("artifacts/targets.csv", "w") as f:
#             f.write(y_train.name)
#         mlflow.log_artifact("artifacts/targets.csv")

#         # Infer signature for the model
#         signature = infer_signature(X_test, y_pred_test_proba)

#         # Log the model with the signature
#         mlflow.keras.log_model(model, "model", signature=signature)

#         # Log classification report and confusion matrix
#         mlflow.log_dict(report, "classification_report.json")
#         mlflow.log_dict({'confusion_matrix': confusion.tolist()}, "confusion_matrix.json")

# def main():
#     mlflow.set_tracking_uri("http://localhost:5000")
#     experiment_name = "experiment_water_quality_mlp"
#     experiment_id = mlflow.create_experiment(experiment_name)
#     client = mlflow.tracking.MlflowClient()
#     experiment = client.get_experiment(experiment_id)

#     data_path = 'data/water_potability.csv'
#     repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
#     version = 'v1'

#     data = load_data(data_path, repo, version)
#     X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_data(data)
#     best_params = tune_hyperparameters(X_train, y_train)
#     final_model = train_final_model(X_train, y_train, X_val, y_val, best_params)
#     evaluate_and_log_model(final_model, X_test, y_test, feature_names, experiment_id, data_path, version, y_train, y_val, best_params)

# if __name__ == "__main__":
#     main()



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
