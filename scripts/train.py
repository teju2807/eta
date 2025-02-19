import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pycaret.regression import *
import mlflow
import os

# Define model directory
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with actual server URL
mlflow.set_experiment("pycaret_distance_prediction")

def train_model():
    # Read the dataset
    df = pd.read_excel('data/old_eta_Data_Set.xlsx')
    
    # Convert categorical features to string
    # df[['SOURCE_CLUSTER', 'DESTI_CLUSTER', 'TRUCK_TYPE']] = df[['SOURCE_CLUSTER', 'DESTI_CLUSTER', 'TRUCK_TYPE']].astype(str)
    
    # Features and target
    X = df[['SOURCE_CLUSTER', 'DESTI_CLUSTER', 'MONTH', 'HOUR', 'TRUCK_TYPE']]
    y = df['DISTANCE']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    with mlflow.start_run():
        # Setup PyCaret
        reg = setup(data=pd.concat([X_train, y_train], axis=1), target='DISTANCE', session_id=123, verbose=True)
        
        # Train best model
        best_model = compare_models()
        tuned_model = tune_model(best_model)
        final_model = finalize_model(tuned_model)
        
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
        
        # Save model correctly
        save_model(final_model, os.path.join(model_dir, 'pycaret_model'))
    
    return {"MAE": mae, "MSE": mse, "R2": r2}

# Train model
train_model()
