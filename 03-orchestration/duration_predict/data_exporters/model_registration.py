import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

@data_exporter
def find_and_register_best_model(model_data, *args, **kwargs):
    """
    Find the best model based on metrics and register it
    """
    # Set up MLflow client
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    # Get experiment
    experiment_name = "taxi-duration-prediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Experiment '{experiment_name}' not found!")
        return None
    
    print(f"Found experiment: {experiment.name} (ID: {experiment.experiment_id})")
    
    # Search all runs in the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],  # Order by RMSE ascending (lower is better)
        max_results=10
    )
    
    if not runs:
        print("No runs found in the experiment!")
        return None
    
    print(f"Found {len(runs)} runs")
    
    # Display all runs with their metrics
    print("\n=== All Runs Summary ===")
    for i, run in enumerate(runs):
        metrics = run.data.metrics
        params = run.data.params
        print(f"\nRun {i+1}: {run.info.run_id}")
        
        # Safe formatting for metrics - check if they exist and are numeric
        rmse = metrics.get('rmse')
        r2_score = metrics.get('r2_score')
        mse = metrics.get('mse')
        
        if rmse is not None:
            print(f"  RMSE: {rmse:.4f}")
        else:
            print(f"  RMSE: N/A")
            
        if r2_score is not None:
            print(f"  R2 Score: {r2_score:.4f}")
        else:
            print(f"  R2 Score: N/A")
            
        if mse is not None:
            print(f"  MSE: {mse:.4f}")
        else:
            print(f"  MSE: N/A")
            
        print(f"  Features: {params.get('n_features', 'N/A')}")
    
    # Filter runs that have RMSE metric for proper comparison
    valid_runs = [run for run in runs if run.data.metrics.get('rmse') is not None]
    
    if not valid_runs:
        print("No runs with RMSE metric found!")
        return None
    
    # Select best run (first one with lowest RMSE)
    best_run = valid_runs[0]
    best_run_id = best_run.info.run_id
    best_metrics = best_run.data.metrics
    
    print(f"\n=== Best Model Selected ===")
    print(f"Run ID: {best_run_id}")
    print(f"RMSE: {best_metrics.get('rmse', 0):.4f}")
    print(f"R2 Score: {best_metrics.get('r2_score', 0):.4f}")
    
    # Register the best model
    model_name = "taxi-duration-best-model"
    model_uri = f"runs:/{best_run_id}/model"
    
    try:
        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags={
                "best_rmse": str(best_metrics.get('rmse', 'N/A')),
                "best_r2": str(best_metrics.get('r2_score', 'N/A')),
                "selection_date": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        )
        
        print(f"\n=== Model Registration Successful ===")
        print(f"Model Name: {model_name}")
        print(f"Version: {model_version.version}")
        print(f"Model URI: {model_uri}")
        
        # Transition model to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        print(f"Model transitioned to 'Staging' stage")
        
        # Try to get model size from artifacts
        print(f"\n=== Model Artifacts ===")
        try:
            artifacts = client.list_artifacts(best_run_id, "model")
            total_size = 0
            for artifact in artifacts:
                if hasattr(artifact, 'file_size') and artifact.file_size:
                    total_size += artifact.file_size
                    print(f"  {artifact.path}: {artifact.file_size} bytes")
            
            if total_size > 0:
                print(f"Total model size: {total_size} bytes")
        except Exception as e:
            print(f"Could not get artifact sizes: {e}")
        
