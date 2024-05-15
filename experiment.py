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
import pandas as pd
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow import keras

# Load the dataset
data = pd.read_csv("data\water_potability.csv")

# Handle missing values (e.g., impute with mean for numerical columns)
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])


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

        # Log model and metrics
        signature = infer_signature(X_val, y_pred)
        mlflow.sklearn.log_model(model, f"model_{algorithm}", signature=signature)
        mlflow.log_metric(f"accuracy_{algorithm}", accuracy)
        mlflow.log_metric(f"precision_{algorithm}", precision)
        mlflow.log_metric(f"recall_{algorithm}", recall)
