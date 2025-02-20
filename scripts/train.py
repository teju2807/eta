import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pycaret.regression import *
import mlflow
import os
import time

# Define model directory
model_dir = 'ETA_MODELS'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Set MLflow tracking URI
mlflow.set_tracking_uri("https://745f-103-246-243-242.ngrok-free.app")  # Replace with actual server URL
print("3333333333333333333333333333333333")
mlflow.set_experiment("pycaret_model_predictions")
print("2222222222222222222222222222")

def train_distance_model(df):
    start_time = time.time()  # Track execution time

    # Features and target
    X = df[['SOURCE_CLUSTER', 'DESTI_CLUSTER', 'DAY', 'MONTH', 'HOUR', 'TRUCK_TYPE']]
    y = df['DISTANCE']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    with mlflow.start_run():
        # Log dataset details
        mlflow.log_param("Train Size", X_train.shape)
        mlflow.log_param("Test Size", X_test.shape)
        
        # Setup PyCaret
        reg = setup(data=pd.concat([X_train, y_train], axis=1), target='DISTANCE', session_id=123, verbose=True)
        
        # Train best model
        best_model = compare_models()
        tuned_model = tune_model(best_model)
        final_model = finalize_model(tuned_model)
        
        # Log best model name
        mlflow.log_param("Best Model", str(best_model))
        
        # Log model hyperparameters
        model_params = tuned_model.get_params()
        for param, value in model_params.items():
            mlflow.log_param(f"Param_{param}", value)

        # Log model to MLflow
        mlflow.sklearn.log_model(final_model, artifact_path="pycaret_model")
        
        # Predict
        predictions = predict_model(final_model, data=X_test)

        # Evaluate
        mae = mean_absolute_error(y_test, predictions['prediction_label'])
        mse = mean_squared_error(y_test, predictions['prediction_label'])
        r2 = r2_score(y_test, predictions['prediction_label'])

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)

        # Log sample predictions
        sample_preds = predictions[['prediction_label']].head(5).to_dict()
        mlflow.log_dict(sample_preds, "distance_model_predictions.json")

        # Log execution time
        execution_time = time.time() - start_time
        mlflow.log_metric("Execution Time (seconds)", execution_time)
        
        # Save model correctly
        save_model(final_model, os.path.join(model_dir, 'distance_model'))
        

def train_speed_model(df):
    start_time = time.time()  # Track execution time

    # Features and target
    X = df[['SOURCE_CLUSTER', 'DESTI_CLUSTER', 'DAY', 'HOUR', 'TRUCK_TYPE']]
    y = df['AVG_SPEED']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    with mlflow.start_run():
        # Log dataset details
        mlflow.log_param("Train Size", X_train.shape)
        mlflow.log_param("Test Size", X_test.shape)
        
        # Setup PyCaret
        reg = setup(data=pd.concat([X_train, y_train], axis=1), target='AVG_SPEED', session_id=123, verbose=True)
        
        # Train best model
        best_model = compare_models()
        tuned_model = tune_model(best_model)
        final_model = finalize_model(tuned_model)
        
        # Log best model name
        mlflow.log_param("Best Model", str(best_model))
        
        # Log model hyperparameters
        model_params = tuned_model.get_params()
        for param, value in model_params.items():
            mlflow.log_param(f"Param_{param}", value)

        # Log model to MLflow
        mlflow.sklearn.log_model(final_model, artifact_path="pycaret_model")
        
        # Predict
        predictions = predict_model(final_model, data=X_test)

        # Evaluate
        mae = mean_absolute_error(y_test, predictions['prediction_label'])
        mse = mean_squared_error(y_test, predictions['prediction_label'])
        r2 = r2_score(y_test, predictions['prediction_label'])

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)

        # Log sample predictions
        sample_preds = predictions[['prediction_label']].head(5).to_dict()
        mlflow.log_dict(sample_preds, "speed_model_predictions.json")

        # Log execution time
        execution_time = time.time() - start_time
        mlflow.log_metric("Execution Time (seconds)", execution_time)
        
        # Save model correctly
        save_model(final_model, os.path.join(model_dir, 'speed_model'))
        
        
def train_duration_model(df):
    start_time = time.time()  # Track execution time

    # Features and target
    X = df[['SOURCE_CLUSTER', 'DESTI_CLUSTER', 'DAY', 'HOUR', 'TRUCK_TYPE','DISTANCE','AVG_SPEED']]
    y = df['DURATION']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    with mlflow.start_run():
        # Log dataset details
        mlflow.log_param("Train Size", X_train.shape)
        mlflow.log_param("Test Size", X_test.shape)
        
        # Setup PyCaret
        reg = setup(data=pd.concat([X_train, y_train], axis=1), target='DURATION', session_id=123, verbose=True)
        
        # Train best model
        best_model = compare_models()
        tuned_model = tune_model(best_model)
        final_model = finalize_model(tuned_model)
        
        # Log best model name
        mlflow.log_param("Best Model", str(best_model))
        
        # Log model hyperparameters
        model_params = tuned_model.get_params()
        for param, value in model_params.items():
            mlflow.log_param(f"Param_{param}", value)

        # Log model to MLflow
        mlflow.sklearn.log_model(final_model, artifact_path="pycaret_model")
        
        # Predict
        predictions = predict_model(final_model, data=X_test)

        # Evaluate
        mae = mean_absolute_error(y_test, predictions['prediction_label'])
        mse = mean_squared_error(y_test, predictions['prediction_label'])
        r2 = r2_score(y_test, predictions['prediction_label'])

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)

        # Log sample predictions
        sample_preds = predictions[['prediction_label']].head(5).to_dict()
        mlflow.log_dict(sample_preds, "duration_model_predictions.json")

        # Log execution time
        execution_time = time.time() - start_time
        mlflow.log_metric("Execution Time (seconds)", execution_time)
        
        # Save model correctly
        save_model(final_model, os.path.join(model_dir, 'duration_model'))


# Read the dataset
df = pd.read_excel('data/old_eta_Data_Set.xlsx')
print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

# Train model
train_distance_model(df)
train_speed_model(df)
train_duration_model(df)