# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from mlflow.models.signature import infer_signature
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestClassifier

# # Load the dataset
# data = pd.read_csv("data\water_potability.csv")

# # Handle missing values (e.g., impute with mean for numerical columns)
# numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
# imputer = SimpleImputer(strategy='mean')
# data[numeric_cols] = imputer.fit_transform(data[numeric_cols])



# # Split the dataset into training and validation sets
# X = data.drop('Potability', axis=1)
# y = data['Potability']
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# # Define the train_model function
# def train_model(X_train, y_train):
#     # Create a Random Forest Classifier
#     rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

#     # Train the model
#     rf_model.fit(X_train, y_train)

#     return rf_model

# # Set up an MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# # Initialize an MLflow experiment
# experiment_name = "experiment_water_quality"
# experiment_id = mlflow.create_experiment(experiment_name)
# client = mlflow.tracking.MlflowClient()
# experiment = client.get_experiment(experiment_id)

# # Start the MLflow experiment
# with mlflow.start_run(experiment_id=experiment_id):
#     # Train a binary classification model
#     model = train_model(X_train, y_train)

#     # Make predictions on the validation set
#     y_pred = model.predict(X_val)

#     # Calculate evaluation metrics
#     accuracy = accuracy_score(y_val, y_pred)
#     precision = precision_score(y_val, y_pred)
#     recall = recall_score(y_val, y_pred)

#     # Log model and metrics
#     signature = infer_signature(X_val, y_pred)
#     mlflow.sklearn.log_model(model, "model", signature=signature)
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)


import mlflow
import dvc.api
import pandas as pd
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
# import tensorflow as tf
from tensorflow import keras


path='data/water_potability.csv'
repo= r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'
version='v1'

data_url = dvc.api.get_url(
  path=path,
  repo=repo,
  rev=version
  )

# Load the dataset
data = pd.read_csv(data_url)

#log data params
# mlflow.log_param('data_url', data_url)
# mlflow.log_param('data_version', version)
# mlflow.log_param('input_rows', data.shape[0])
# mlflow.log_param('input_cols', data.shape[1])


# Handle missing values (e.g., impute with mean for numerical columns)
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

#to do: train, val, test
# Split the dataset into training and validation sets
X = data.drop('Potability', axis=1)
y = data['Potability']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the train_model function
def train_model(X_train, y_train, algorithm):
    """
    Train a binary classification model based on the specified algorithm.

    Args:
        X_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training labels.
        algorithm (str): Algorithm to use for training the model.
            Accepted values: "random_forest", "xgboost", "mlp"

    Returns:
        Trained model object.
    """

    #to-do : add params
    if algorithm == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm == "xgboost":
        # Import XGBClassifier from the xgboost library
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, random_state=42)
    elif algorithm == "mlp":
        # Define the MLP model architecture
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model with backpropagation
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    else:
        raise ValueError("Invalid algorithm specified")

    # Train the model
    if algorithm == "mlp":
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
    else:
        model.fit(X_train, y_train)

    return model

# Set up an MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Initialize an MLflow experiment
experiment_name = "experiment_water_quality"
experiment_id = mlflow.create_experiment(experiment_name)
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment(experiment_id)

# Train and log models for different algorithms
for algorithm in ["random_forest", "xgboost", "mlp"]:
    with mlflow.start_run(experiment_id=experiment_id):
        # Train the model
        model = train_model(X_train, y_train, algorithm)

        # Make predictions on the validation set
        if algorithm == "mlp":
            y_pred = (model.predict(X_val) > 0.5).astype(int)
        else:
            y_pred = model.predict(X_val)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        # #log artifacts: columns used for modeling
        # cols_x = pd.DataFrame(list(X_train.columns))
        # cols_x.to_csv('features.csv', header=False, index=False)
        # mlflow.log_artifact('features.csv')


        # # Extract the name of the Series
        # series_name = y_train.name
        # cols_y = pd.DataFrame({series_name: ['Potability']})
        # cols_y.to_csv('targets.csv', header=False, index=False)
        # mlflow.log_artifact('targets.csv')

        
        # Log artifacts: columns used for modeling
        X_train.columns.to_series().to_csv('features.csv', header=False, index=False)
        mlflow.log_artifact('features.csv')

        # Extract the name of the Series
        series_name = y_train.name

        # Log the name of the target column
        with open('targets.csv', 'w') as f:
            f.write(series_name)
        mlflow.log_artifact('targets.csv')

        # Log model and metrics
        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.log_model(model, f"model_{algorithm}", signature=signature)
        mlflow.log_metric(f"accuracy_{algorithm}", accuracy)
        mlflow.log_metric(f"precision_{algorithm}", precision)
        mlflow.log_metric(f"recall_{algorithm}", recall)

        #log data params
        mlflow.log_param('data_url', data_url)
        mlflow.log_param('data_version', version)
        mlflow.log_param('input_rows', data.shape[0])
        mlflow.log_param('input_cols', data.shape[1])



# import mlflow
# import dvc.api
# import pandas as pd
# import mlflow.sklearn
# from mlflow.models.signature import infer_signature
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from tensorflow import keras
# from scipy.stats import randint
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer

# path = 'data/water_potability.csv'
# repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'

# data_url = dvc.api.get_url(
#     path=path,
#     repo=repo
# )

# # Load the dataset
# data = pd.read_csv(data_url)

# # Handle missing values (e.g., impute with mean for numerical columns)
# numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
# imputer = SimpleImputer(strategy='mean')
# data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# # Split the dataset into train, validation, and test sets
# X = data.drop('Potability', axis=1)
# y = data['Potability']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Define the preprocessing steps
# numeric_transformer = StandardScaler()
# preprocessor = ColumnTransformer(transformers=[
#     ('num', numeric_transformer, X_train.columns)
# ])

# # Set up an MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# # Initialize an MLflow experiment
# experiment_name = "experiment_water_quality"
# experiment_id = mlflow.create_experiment(experiment_name)
# client = mlflow.tracking.MlflowClient()
# experiment = client.get_experiment(experiment_id)

# # Hyperparameter tuning and model selection
# best_model = None
# best_algorithm = None
# best_params = None
# best_accuracy = 0

# for algorithm, model_class in [("random_forest", RandomForestClassifier), ("mlp", None)]:
#     if algorithm == "random_forest":
#         random_forest_param_dist = {
#             "n_estimators": randint(100, 500),
#             "max_depth": randint(5, 20),
#             "min_samples_split": randint(2, 10),
#             "min_samples_leaf": randint(1, 5),
#             "max_features": randint(1, X_train.shape[1]),
#             "random_state": [42]
#         }

#         pipeline = Pipeline([
#             ('preprocessor', preprocessor),
#             ('classifier', model_class())
#         ])

#         random_search = RandomizedSearchCV(pipeline, param_distributions=random_forest_param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
#         random_search.fit(X_train, y_train)
#         best_params = random_search.best_params_

#         with mlflow.start_run(experiment_id=experiment_id):
#             # Train the final model with the best hyperparameters
#             final_model = random_search.best_estimator_

#             # Evaluate the final model on the validation set
#             y_pred = final_model.predict(X_val)
#             accuracy = accuracy_score(y_val, y_pred)
#             precision = precision_score(y_val, y_pred)
#             recall = recall_score(y_val, y_pred)

#             # Log model and metrics
#             signature = infer_signature(X_val, y_pred)
#             mlflow.sklearn.log_model(final_model, f"model_{algorithm}", signature=signature)
#             mlflow.log_params(best_params)
#             mlflow.log_metric(f"accuracy_{algorithm}", accuracy)
#             mlflow.log_metric(f"precision_{algorithm}", precision)
#             mlflow.log_metric(f"recall_{algorithm}", recall)

#             if accuracy > best_accuracy:
#                 best_model = final_model
#                 best_algorithm = algorithm
#                 best_params = best_params
#                 best_accuracy = accuracy

#     elif algorithm == "mlp":
#         mlp_param_dist = {
#             "epochs": randint(20, 100),
#             "batch_size": randint(16, 64),
#             "optimizer": ["adam", "rmsprop"],
#             "learning_rate": [0.001, 0.005, 0.01],
#             "dropout_rate": randint(2, 5) / 10
#         }

#         def create_model(params):
#             model = keras.Sequential([
#                 keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#                 keras.layers.Dropout(params["dropout_rate"]),
#                 keras.layers.Dense(32, activation='relu'),
#                 keras.layers.Dropout(params["dropout_rate"]),
#                 keras.layers.Dense(1, activation='sigmoid')
#             ])

#             optimizer = keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["optimizer"] == "adam" else keras.optimizers.RMSprop(learning_rate=params["learning_rate"])
#             model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#             return model

#         random_search = RandomizedSearchCV(create_model, param_distributions=mlp_param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
#         random_search.fit(X_train, y_train)
#         best_params = random_search.best_params_

#         with mlflow.start_run(experiment_id=experiment_id):
#             # Train the final model with the best hyperparameters
#             final_model = create_model(best_params)
#             final_model.fit(X_train, y_train, **best_params)

#             # Evaluate the final model on the validation set
#             y_pred = (final_model.predict(X_val) > 0.5).astype(int)
#             accuracy = accuracy_score(y_val, y_pred)
#             precision = precision_score(y_val, y_pred)
#             recall = recall_score(y_val, y_pred)

#             # Log model and metrics
#             signature = infer_signature(X_val, y_pred)
#             mlflow.keras.log_model(final_model, f"model_{algorithm}", signature=signature)
#             mlflow.log_params(best_params)
#             mlflow.log_metric(f"accuracy_{algorithm}", accuracy)
#             mlflow.log_metric(f"precision_{algorithm}", precision)
#             mlflow.log_metric(f"recall_{algorithm}", recall)

#             if accuracy > best_accuracy:
#                 best_model = final_model
#                 best_algorithm = algorithm
#                 best_params = best_params
#                 best_accuracy = accuracy

# # Register the best model in the MLflow model registry
# if best_model is not None:
#     model_name = "water_potability_model"
#     mlflow.sklearn.log_model(best_model, "models/{}/production".format(model_name))

#     # Get the latest production model from the registry
#     model_uri = "models:/{}/production".format(model_name)
#     production_model = mlflow.sklearn.load_model(model_uri)

#     # Evaluate the best model on the test set
#     y_pred = production_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)



# import mlflow
# import dvc.api
# import pandas as pd
# import mlflow.sklearn
# from mlflow.models.signature import infer_signature
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from tensorflow import keras
# from scipy.stats import randint
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer

# path = 'data/water_potability.csv'
# repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'

# data_url = dvc.api.get_url(
#     path=path,
#     repo=repo
# )

# # Load the dataset
# data = pd.read_csv(data_url)

# # Handle missing values (e.g., impute with mean for numerical columns)
# numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
# imputer = SimpleImputer(strategy='mean')
# data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# # Split the dataset into train, validation, and test sets
# X = data.drop('Potability', axis=1)
# y = data['Potability']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Define the preprocessing steps
# numeric_transformer = StandardScaler()
# preprocessor = ColumnTransformer(transformers=[
#     ('num', numeric_transformer, X_train.columns)
# ])

# # Set up an MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# # Initialize an MLflow experiment
# experiment_name = "experiment_water_quality"
# experiment_id = mlflow.create_experiment(experiment_name)
# client = mlflow.tracking.MlflowClient()
# experiment = client.get_experiment(experiment_id)

# # Hyperparameter tuning and model selection
# best_model = None
# best_algorithm = None
# best_params = None
# best_accuracy = 0

# for algorithm, model_class in [("random_forest", RandomForestClassifier), ("mlp", None)]:
#     if algorithm == "random_forest":
#         random_forest_param_dist = {
#             "classifier__n_estimators": randint(100, 500),
#             "classifier__max_depth": randint(5, 20),
#             "classifier__min_samples_split": randint(2, 10),
#             "classifier__min_samples_leaf": randint(1, 5),
#             "classifier__max_features": randint(1, X_train.shape[1]),
#             "classifier__random_state": [42]
#         }

#         pipeline = Pipeline([
#             ('preprocessor', preprocessor),
#             ('classifier', model_class())
#         ])

#         random_search = RandomizedSearchCV(pipeline, param_distributions=random_forest_param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
#         random_search.fit(X_train, y_train)
#         best_params = random_search.best_params_

#         with mlflow.start_run(experiment_id=experiment_id):
#             # Train the final model with the best hyperparameters
#             final_model = random_search.best_estimator_

#             # Evaluate the final model on the validation set
#             y_pred = final_model.predict(X_val)
#             accuracy = accuracy_score(y_val, y_pred)
#             precision = precision_score(y_val, y_pred)
#             recall = recall_score(y_val, y_pred)

#             # Log model and metrics
#             signature = infer_signature(X_val, y_pred)
#             mlflow.sklearn.log_model(final_model, f"model_{algorithm}", signature=signature)
#             mlflow.log_params(best_params)
#             mlflow.log_metric(f"accuracy_{algorithm}", accuracy)
#             mlflow.log_metric(f"precision_{algorithm}", precision)
#             mlflow.log_metric(f"recall_{algorithm}", recall)

#             if accuracy > best_accuracy:
#                 best_model = final_model
#                 best_algorithm = algorithm
#                 best_params = best_params
#                 best_accuracy = accuracy

#     elif algorithm == "mlp":
#         mlp_param_dist = {
#             "epochs": randint(20, 100),
#             "batch_size": randint(16, 64),
#             "optimizer": ["adam", "rmsprop"],
#             "learning_rate": [0.001, 0.005, 0.01],
#             "dropout_rate": randint(2, 5).rvs(1)[0] / 10
#         }

#         def create_model(params):
#             model = keras.Sequential([
#                 keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#                 keras.layers.Dropout(params["dropout_rate"]),
#                 keras.layers.Dense(32, activation='relu'),
#                 keras.layers.Dropout(params["dropout_rate"]),
#                 keras.layers.Dense(1, activation='sigmoid')
#             ])

#             optimizer = keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["optimizer"] == "adam" else keras.optimizers.RMSprop(learning_rate=params["learning_rate"])
#             model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#             return model

#         random_search = RandomizedSearchCV(create_model, param_distributions=mlp_param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
#         random_search.fit(X_train, y_train)
#         best_params = random_search.best_params_

#         with mlflow.start_run(experiment_id=experiment_id):
#             # Train the final model with the best hyperparameters
#             final_model = create_model(best_params)
#             final_model.fit(X_train, y_train, **best_params)

#             # Evaluate the final model on the validation set
#             y_pred = (final_model.predict(X_val) > 0.5).astype(int)
#             accuracy = accuracy_score(y_val, y_pred)
#             precision = precision_score(y_val, y_pred)
#             recall = recall_score(y_val, y_pred)

#             # Log model and metrics
#             signature = infer_signature(X_val, y_pred)
#             mlflow.keras.log_model(final_model, f"model_{algorithm}", signature=signature)
#             mlflow.log_params(best_params)
#             mlflow.log_metric(f"accuracy_{algorithm}", accuracy)
#             mlflow.log_metric(f"precision_{algorithm}", precision)
#             mlflow.log_metric(f"recall_{algorithm}", recall)

#             if accuracy > best_accuracy:
#                 best_model = final_model
#                 best_algorithm = algorithm
#                 best_params = best_params
#                 best_accuracy = accuracy

# # Register the best model in the MLflow model registry
# if best_model is not None:
#     model_name = "water_potability_model"
#     mlflow.sklearn.log_model(best_model, "models/{}/production".format(model_name))

#     # Get the latest production model from the registry
#     model_uri = "models:/{}/production".format(model_name)
#     production_model = mlflow.sklearn.load_model(model_uri)

#     # Evaluate the best model on the test set
#     y_pred = production_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)

#     # Log the final model and metrics
#     signature = infer_signature(X_test, y_pred)
#     mlflow.sklearn.log_model(production_model, "final_model", signature=signature)
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)



# import mlflow
# import dvc.api
# import pandas as pd
# import mlflow.sklearn
# from mlflow.models.signature import infer_signature
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from tensorflow import keras
# from scipy.stats import randint
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer

# path = 'data/water_potability.csv'
# repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'

# data_url = dvc.api.get_url(
#     path=path,
#     repo=repo
# )

# # Load the dataset
# data = pd.read_csv(data_url)

# # Handle missing values (e.g., impute with mean for numerical columns)
# numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
# imputer = SimpleImputer(strategy='mean')
# data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# # Split the dataset into train, validation, and test sets
# X = data.drop('Potability', axis=1)
# y = data['Potability']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Define the preprocessing steps
# numeric_transformer = StandardScaler()
# preprocessor = ColumnTransformer(transformers=[
#     ('num', numeric_transformer, X_train.columns)
# ])

# # Set up an MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# # Initialize an MLflow experiment
# experiment_name = "experiment_water_quality"
# experiment_id = mlflow.create_experiment(experiment_name)
# client = mlflow.tracking.MlflowClient()
# experiment = client.get_experiment(experiment_id)

# # Hyperparameter tuning and model selection
# best_model = None
# best_algorithm = None
# best_params = None
# best_accuracy = 0

# for algorithm, model_class in [("random_forest", RandomForestClassifier), ("mlp", None)]:
#     if algorithm == "random_forest":
#         random_forest_param_dist = {
#             "classifier__n_estimators": randint(100, 500),
#             "classifier__max_depth": randint(5, 20),
#             "classifier__min_samples_split": randint(2, 10),
#             "classifier__min_samples_leaf": randint(1, 5),
#             "classifier__max_features": randint(1, X_train.shape[1]),
#             "classifier__random_state": [42]
#         }

#         pipeline = Pipeline([
#             ('preprocessor', preprocessor),
#             ('classifier', model_class())
#         ])

#         random_search = RandomizedSearchCV(pipeline, param_distributions=random_forest_param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
#         random_search.fit(X_train, y_train)
#         best_params = random_search.best_params_

#         with mlflow.start_run(experiment_id=experiment_id):
#             # Train the final model with the best hyperparameters
#             final_model = random_search.best_estimator_

#             # Evaluate the final model on the validation set
#             y_pred = final_model.predict(X_val)
#             accuracy = accuracy_score(y_val, y_pred)
#             precision = precision_score(y_val, y_pred)
#             recall = recall_score(y_val, y_pred)

#             # Log model and metrics
#             signature = infer_signature(X_val, y_pred)
#             mlflow.sklearn.log_model(final_model, f"model_{algorithm}", signature=signature)
#             mlflow.log_params(best_params)
#             mlflow.log_metric(f"accuracy_{algorithm}", accuracy)
#             mlflow.log_metric(f"precision_{algorithm}", precision)
#             mlflow.log_metric(f"recall_{algorithm}", recall)

#         if accuracy > best_accuracy:
#             best_model = final_model
#             best_algorithm = algorithm
#             best_params = best_params
#             best_accuracy = accuracy

#         elif algorithm == "mlp":
#             mlp_param_dist = {
#             "epochs": randint(20, 100),
#             "batch_size": randint(16, 64),
#             "optimizer": ["adam", "rmsprop"],
#             "learning_rate": [0.001, 0.005, 0.01],
#             "dropout_rate": randint(2, 5).rvs(1)[0] / 10
#             }

#             def create_model(params):
#                 model = keras.Sequential([
#                     keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#                     keras.layers.Dropout(params["dropout_rate"]),
#                     keras.layers.Dense(32, activation='relu'),
#                     keras.layers.Dropout(params["dropout_rate"]),
#                     keras.layers.Dense(1, activation='sigmoid')
#                 ])

#                 optimizer = keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["optimizer"] == "adam" else keras.optimizers.RMSprop(learning_rate=params["learning_rate"])
#                 model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#                 return model

#             random_search = RandomizedSearchCV(create_model, param_distributions=mlp_param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
#             random_search.fit(X_train, y_train)
#             best_params = random_search.best_params_

#             with mlflow.start_run(experiment_id=experiment_id):
#                 # Train the final model with the best hyperparameters
#                 final_model = create_model(best_params)
#                 final_model.fit(X_train, y_train, **best_params)

#                 # Evaluate the final model on the validation set
#                 y_pred = (final_model.predict(X_val) > 0.5).astype(int)
#                 accuracy = accuracy_score(y_val, y_pred)
#                 precision = precision_score(y_val, y_pred)
#                 recall = recall_score(y_val, y_pred)

#                 # Log model and metrics
#                 signature = infer_signature(X_val, y_pred)
#                 mlflow.keras.log_model(final_model, f"model_{algorithm}", signature=signature)
#                 mlflow.log_params(best_params)
#                 mlflow.log_metric(f"accuracy_{algorithm}", accuracy)
#                 mlflow.log_metric(f"precision_{algorithm}", precision)
#                 mlflow.log_metric(f"recall_{algorithm}", recall)

#                 if accuracy > best_accuracy:
#                     best_model = final_model
#                     best_algorithm = algorithm
#                     best_params = best_params
#                     best_accuracy = accuracy

# # Register the best model in the MLflow model registry
# if best_model is not None:
#     model_name = "water_potability_model"
#     mlflow.sklearn.log_model(best_model, "models/{}/production".format(model_name))

#     # Get the latest production model from the registry
#     model_uri = "models:/{}/production".format(model_name)
#     production_model = mlflow.sklearn.load_model(model_uri)

#     # Evaluate the best model on the test set
#     y_pred = production_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)

#     # Log the final model and metrics
#     signature = infer_signature(X_test, y_pred)
#     mlflow.sklearn.log_model(production_model, "final_model", signature=signature)
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)






# import mlflow
# import dvc.api
# import pandas as pd
# import mlflow.sklearn
# from mlflow.models.signature import infer_signature
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from scipy.stats import randint, uniform
# from keras.wrappers.scikit_learn import KerasClassifier
# import tensorflow as tf
# import joblib

# path = 'data/water_potability.csv'
# repo = r'C:\Users\eakli\Downloads\task\ecole\mlops-mlflow'

# data_url = dvc.api.get_url(
#     path=path,
#     repo=repo
# )

# # Load the dataset
# data = pd.read_csv(data_url)

# # Handle missing values (e.g., impute with mean for numerical columns)
# numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
# imputer = SimpleImputer(strategy='mean')
# data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# # Split the dataset into train, validation, and test sets
# X = data.drop('Potability', axis=1)
# y = data['Potability']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Define the preprocessing steps
# numeric_transformer = StandardScaler()
# preprocessor = ColumnTransformer(transformers=[
#     ('num', numeric_transformer, X_train.columns)
# ])

# # Set up an MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# # Initialize an MLflow experiment
# experiment_name = "experiment_water_quality"
# mlflow.set_experiment(experiment_name)

# # Hyperparameter tuning and model selection
# best_model = None
# best_algorithm = None
# best_params = None
# best_accuracy = 0

# for algorithm, model_class in [("random_forest", RandomForestClassifier), ("mlp", None)]:
#     if algorithm == "random_forest":
#         random_forest_param_dist = {
#             "n_estimators": randint(100, 500),
#             "max_depth": randint(5, 20),
#             "min_samples_split": randint(2, 10),
#             "min_samples_leaf": randint(1, 5),
#             "max_features": randint(1, X_train.shape[1]),
#             "random_state": [42]
#         }

#         pipeline = Pipeline([
#             ('preprocessor', preprocessor),
#             ('classifier', model_class())
#         ])

#         random_search = RandomizedSearchCV(pipeline, param_distributions=random_forest_param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
#         random_search.fit(X_train, y_train)
#         best_params = random_search.best_params_

#         with mlflow.start_run():
#             # Train the final model with the best hyperparameters
#             final_model = random_search.best_estimator_

#             # Evaluate the final model on the validation set
#             y_pred = final_model.predict(X_val)
#             accuracy = accuracy_score(y_val, y_pred)
#             precision = precision_score(y_val, y_pred)
#             recall = recall_score(y_val, y_pred)

#             # Log model and metrics
#             signature = infer_signature(X_val, y_pred)
#             mlflow.sklearn.log_model(final_model, f"model_{algorithm}", signature=signature)
#             mlflow.log_params(best_params)
#             mlflow.log_metric(f"accuracy_{algorithm}", accuracy)
#             mlflow.log_metric(f"precision_{algorithm}", precision)
#             mlflow.log_metric(f"recall_{algorithm}", recall)

#             if accuracy > best_accuracy:
#                 best_model = final_model
#                 best_algorithm = algorithm
#                 best_params = best_params
#                 best_accuracy = accuracy

#     elif algorithm == "mlp":
#         mlp_param_dist = {
#             "epochs": randint(20, 100),
#             "batch_size": randint(16, 64),
#             "optimizer": ["adam", "rmsprop"],
#             "learning_rate": uniform(0.001, 0.01)
#         }

#         def create_model(params):
#             model = tf.keras.Sequential([
#                 tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#                 tf.keras.layers.Dropout(0.4),
#                 tf.keras.layers.Dense(32, activation='relu'),
#                 tf.keras.layers.Dropout(0.4),
#                 tf.keras.layers.Dense(1, activation='sigmoid')
#             ])

#             optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]) if params["optimizer"] == "adam" else tf.keras.optimizers.RMSprop(learning_rate=params["learning_rate"])
#             model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

#             return model

#         model = KerasClassifier(build_fn=create_model, verbose=0)
#         random_search = RandomizedSearchCV(model, param_distributions=mlp_param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)
#         random_search.fit(X_train, y_train)
#         best_params = random_search.best_params_

#         with mlflow.start_run():
#             # Train the final model with the best hyperparameters
#             final_model = create_model(best_params)
#             final_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)

#             # Evaluate the final model on the validation set
#             y_pred = (final_model.predict(X_val) > 0.5).astype(int)
#             accuracy = accuracy_score(y_val, y_pred)
#             precision = precision_score(y_val, y_pred)
#             recall = recall_score(y_val, y_pred)

#             # Log model and metrics
#             signature = infer_signature(X_val, y_pred)
#             mlflow.keras.log_model(final_model, f"model_{algorithm}", signature=signature)
#             mlflow.log_params(best_params)
#             mlflow.log_metric(f"accuracy_{algorithm}", accuracy)
#             mlflow.log_metric(f"precision_{algorithm}", precision)
#             mlflow.log_metric(f"recall_{algorithm}", recall)

#             if accuracy > best_accuracy:
#                 best_model = final_model
#                 best_algorithm = algorithm
#                 best_params = best_params
#                 best_accuracy = accuracy

# # Register the best model in the MLflow model registry
# if best_model is not None:
#     model_name = "water_potability_model"
#     mlflow.sklearn.log_model(best_model, "models/{}/production".format(model_name))

#     # Get the latest production model from the registry
#     model_uri = "models:/{}/production".format(model_name)
#     production_model = mlflow.sklearn.load_model(model_uri)

#     # Evaluate the best model on the test set
#     y_pred = production_model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)

#     # Log the final model and metrics
#     signature = infer_signature(X_test, y_pred)
#     mlflow.sklearn.log_model(production_model, "final_model", signature=signature)
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)

