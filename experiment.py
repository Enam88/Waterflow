


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



