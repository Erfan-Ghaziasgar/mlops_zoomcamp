import pandas as pd
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

@transformer
def train_model(df, *args, **kwargs):
    """
    Train linear regression model and log metrics to MLflow
    """
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("taxi-duration-prediction")
    
    with mlflow.start_run() as run:
        # Prepare features - keep PU and DO locations separate
        categorical = ['PULocationID', 'DOLocationID']
        
        # Convert to dictionary format
        dicts = df[categorical].to_dict(orient='records')
        
        # Fit DictVectorizer
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
        
        # Target variable
        y = df['duration'].values
        
        # Train linear regression with default parameters
        lr = LinearRegression()
        lr.fit(X, y)
        
        # Make predictions for metric calculation
        y_pred = lr.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Log parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", "PULocationID,DOLocationID")
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", X.shape[1])
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mean_duration", np.mean(y))
        mlflow.log_metric("std_duration", np.std(y))
        
        # Print intercept (for Question 5)
        print(f"Model intercept: {lr.intercept_:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.2f}")
        
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="model",
            registered_model_name="taxi-duration-model"
        )
        
        # Log the vectorizer as well
        mlflow.sklearn.log_model(dv, "vectorizer")
        
        # Log run info
        print(f"MLflow run ID: {run.info.run_id}")
        
        return {
            'model': lr,
            'vectorizer': dv,
            'intercept': lr.intercept_,
            'rmse': rmse,
            'r2_score': r2,
            'run_id': run.info.run_id
        }