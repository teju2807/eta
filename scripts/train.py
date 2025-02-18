import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pycaret.regression import *
import mlflow
import pickle
import os

# Define the directory where you want to save the model
model_dir = 'models'

# Check if the directory exists, and if not, create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def train_model():
    
    # Set the tracking URI for MLflow (this is where your MLflow server is running)
    mlflow.set_tracking_uri(r"D:\ETA\mlops_project\mlruns")  # Replace with your server URL

    # Read the dataset
    df = pd.read_excel('data/old_eta_Data_Set.xlsx')
    
    # Features and target
    X = df[['SOURCE_CLUSTER', 'DESTI_CLUSTER', 'MONTH', 'HOUR', 'TRUCK_TYPE']]
    y = df['DISTANCE']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    
     # Set the MLflow experiment
    mlflow.set_experiment("pycaret_distance_prediction")

    with mlflow.start_run():
        reg = setup(data=pd.concat([X_train, y_train], axis=1), target='DISTANCE', session_id=123)
        best_model = compare_models()
        tuned_model = tune_model(best_model)
        final_model = finalize_model(tuned_model)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(final_model, artifact_path="pycaret_model")
        
        # Predict on test set
        predictions = predict_model(final_model, data=X_test)
        
        # Evaluate model
        mae = mean_absolute_error(y_test, predictions['prediction_label'])
        mse = mean_squared_error(y_test, predictions['prediction_label'])
        r2 = r2_score(y_test, predictions['prediction_label'])
        
        # Log metrics to MLflow
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)
        
        # Save model to disk
        pickle.dump(final_model, open(os.path.join(model_dir, 'model.pkl'), 'wb'))
    return {"MAE": mae, "MSE": mse, "R2": r2}