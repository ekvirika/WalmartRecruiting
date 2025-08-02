"""
MLflow Configuration with DagsHub Integration
For Walmart Sales Forecasting Project
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import dagshub
import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import tempfile

class MLflowDagsHubLogger:
    """
    Comprehensive MLflow logger with DagsHub integration for time series forecasting
    """
    
    def __init__(self, 
                 dagshub_repo_owner: str,
                 dagshub_repo_name: str,
                 dagshub_token: str = None):
        """
        Initialize MLflow with DagsHub integration
        
        Args:
            dagshub_repo_owner: Your DagsHub username
            dagshub_repo_name: Repository name on DagsHub
            dagshub_token: DagsHub authentication token
        """
        self.dagshub_repo_owner = dagshub_repo_owner
        self.dagshub_repo_name = dagshub_repo_name
        
        # Set up DagsHub
        dagshub.init(repo_owner=dagshub_repo_owner, 
                    repo_name=dagshub_repo_name, 
                    mlflow=True)
        
        # Set MLflow tracking URI to DagsHub
        mlflow_tracking_uri = f"https://dagshub.com/{dagshub_repo_owner}/{dagshub_repo_name}.mlflow"
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Set authentication if token provided
        if dagshub_token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
            os.environ['MLFLOW_TRACKING_PASSWORD'] = ""
        
        self.current_experiment = None
        self.current_run = None
        
    def create_experiment(self, experiment_name: str) -> str:
        """
        Create or get existing MLflow experiment
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            experiment_id: ID of the created/existing experiment
        """
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
        
        self.current_experiment = experiment_name
        mlflow.set_experiment(experiment_name)
        return experiment_id
    
    def start_run(self, run_name: str, nested: bool = False) -> mlflow.ActiveRun:
        """
        Start a new MLflow run
        
        Args:
            run_name: Name of the run
            nested: Whether this is a nested run
            
        Returns:
            Active MLflow run
        """
        self.current_run = mlflow.start_run(run_name=run_name, nested=nested)
        return self.current_run
    
    def log_preprocessing_step(self, 
                              step_name: str,
                              input_shape: tuple,
                              output_shape: tuple,
                              preprocessing_params: Dict[str, Any] = None,
                              data_quality_metrics: Dict[str, Any] = None):
        """
        Log preprocessing step information
        
        Args:
            step_name: Name of preprocessing step
            input_shape: Shape of input data
            output_shape: Shape of output data
            preprocessing_params: Parameters used in preprocessing
            data_quality_metrics: Data quality metrics
        """
        with mlflow.start_run(run_name=f"{step_name}_preprocessing", nested=True):
            # Log shapes
            mlflow.log_param("input_rows", input_shape[0])
            mlflow.log_param("input_cols", input_shape[1] if len(input_shape) > 1 else 1)
            mlflow.log_param("output_rows", output_shape[0])
            mlflow.log_param("output_cols", output_shape[1] if len(output_shape) > 1 else 1)
            
            # Log preprocessing parameters
            if preprocessing_params:
                for key, value in preprocessing_params.items():
                    mlflow.log_param(f"preprocess_{key}", value)
            
            # Log data quality metrics
            if data_quality_metrics:
                for key, value in data_quality_metrics.items():
                    mlflow.log_metric(f"data_quality_{key}", value)
            
            # Log step completion
            mlflow.log_metric("preprocessing_completed", 1)
    
    def log_feature_engineering(self,
                               features_before: list,
                               features_after: list,
                               feature_importance: Dict[str, float] = None,
                               feature_selection_method: str = None):
        """
        Log feature engineering information
        
        Args:
            features_before: List of features before engineering
            features_after: List of features after engineering
            feature_importance: Feature importance scores
            feature_selection_method: Method used for feature selection
        """
        with mlflow.start_run(run_name="feature_engineering", nested=True):
            # Log feature counts
            mlflow.log_param("features_before_count", len(features_before))
            mlflow.log_param("features_after_count", len(features_after))
            
            # Log feature names
            mlflow.log_param("features_before", str(features_before[:20]))  # Limit to first 20
            mlflow.log_param("features_after", str(features_after[:20]))   # Limit to first 20
            
            # Log feature selection method
            if feature_selection_method:
                mlflow.log_param("feature_selection_method", feature_selection_method)
            
            # Log feature importance
            if feature_importance:
                for feature, importance in list(feature_importance.items())[:20]:  # Top 20
                    mlflow.log_metric(f"feature_importance_{feature}", importance)
    
    def log_model_training(self,
                          model,
                          model_params: Dict[str, Any],
                          training_metrics: Dict[str, float],
                          validation_metrics: Dict[str, float],
                          model_name: str,
                          tags: Dict[str, str] = None):
        """
        Log model training information
        
        Args:
            model: Trained model object
            model_params: Model hyperparameters
            training_metrics: Training metrics
            validation_metrics: Validation metrics
            model_name: Name of the model
            tags: Additional tags for the run
        """
        # Log parameters
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        
        # Log training metrics
        for key, value in training_metrics.items():
            mlflow.log_metric(f"train_{key}", value)
        
        # Log validation metrics
        for key, value in validation_metrics.items():
            mlflow.log_metric(f"val_{key}", value)
        
        # Log model
        if hasattr(model, 'predict'):
            if isinstance(model, Pipeline):
                mlflow.sklearn.log_model(model, model_name)
            else:
                # Handle different model types
                if hasattr(model, '__module__'):
                    if 'torch' in model.__module__:
                        mlflow.pytorch.log_model(model, model_name)
                    elif 'sklearn' in model.__module__:
                        mlflow.sklearn.log_model(model, model_name)
                    else:
                        # Generic model logging
                        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
                            joblib.dump(model, tmp.name)
                            mlflow.log_artifact(tmp.name, f"{model_name}.pkl")
        
        # Log tags
        if tags:
            mlflow.set_tags(tags)
    
    def log_cross_validation(self,
                           cv_scores: Dict[str, list],
                           cv_method: str,
                           n_folds: int):
        """
        Log cross-validation results
        
        Args:
            cv_scores: Dictionary of CV scores for each metric
            cv_method: Cross-validation method used
            n_folds: Number of folds
        """
        with mlflow.start_run(run_name="cross_validation", nested=True):
            mlflow.log_param("cv_method", cv_method)
            mlflow.log_param("n_folds", n_folds)
            
            for metric_name, scores in cv_scores.items():
                mlflow.log_metric(f"cv_{metric_name}_mean", np.mean(scores))
                mlflow.log_metric(f"cv_{metric_name}_std", np.std(scores))
                mlflow.log_metric(f"cv_{metric_name}_min", np.min(scores))
                mlflow.log_metric(f"cv_{metric_name}_max", np.max(scores))
                
                # Log individual fold scores
                for i, score in enumerate(scores):
                    mlflow.log_metric(f"cv_{metric_name}_fold_{i+1}", score)
    
    def log_hyperparameter_tuning(self,
                                 best_params: Dict[str, Any],
                                 best_score: float,
                                 tuning_method: str,
                                 search_space: Dict[str, Any] = None,
                                 n_trials: int = None):
        """
        Log hyperparameter tuning results
        
        Args:
            best_params: Best parameters found
            best_score: Best score achieved
            tuning_method: Method used for tuning
            search_space: Search space definition
            n_trials: Number of trials performed
        """
        with mlflow.start_run(run_name="hyperparameter_tuning", nested=True):
            mlflow.log_param("tuning_method", tuning_method)
            if n_trials:
                mlflow.log_param("n_trials", n_trials)
            
            # Log best parameters
            for key, value in best_params.items():
                mlflow.log_param(f"best_{key}", value)
            
            mlflow.log_metric("best_score", best_score)
            
            # Log search space
            if search_space:
                for key, value in search_space.items():
                    mlflow.log_param(f"search_space_{key}", str(value))
    
    def log_time_series_metrics(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               is_holiday: np.ndarray = None,
                               prefix: str = ""):
        """
        Log time series specific metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            is_holiday: Holiday indicator for WMAE calculation
            prefix: Prefix for metric names
        """
        prefix = f"{prefix}_" if prefix else ""
        
        # Standard metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        mlflow.log_metric(f"{prefix}mae", mae)
        mlflow.log_metric(f"{prefix}mse", mse)
        mlflow.log_metric(f"{prefix}rmse", rmse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        mlflow.log_metric(f"{prefix}mape", mape)
        
        # WMAE (Weighted Mean Absolute Error) if holiday data available
        if is_holiday is not None:
            weights = np.where(is_holiday, 5, 1)
            wmae = np.average(np.abs(y_true - y_pred), weights=weights)
            mlflow.log_metric(f"{prefix}wmae", wmae)
    
    def register_best_model(self,
                           model_name: str,
                           model_version: str,
                           model_stage: str = "Staging"):
        """
        Register the best model in MLflow Model Registry
        
        Args:
            model_name: Name for the registered model
            model_version: Version of the model
            model_stage: Stage of the model (Staging, Production, etc.)
        """
        try:
            # Create registered model
            mlflow.register_model(
                model_uri=f"runs:/{self.current_run.info.run_id}/{model_name}",
                name=model_name
            )
            
            # Transition to specified stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=model_stage
            )
            
            print(f"Model {model_name} version {model_version} registered and moved to {model_stage}")
            
        except Exception as e:
            print(f"Error registering model: {e}")
    
    def end_run(self):
        """End the current MLflow run"""
        if self.current_run:
            mlflow.end_run()
            self.current_run = None

# Utility function for WMAE calculation
def calculate_wmae(y_true: np.ndarray, y_pred: np.ndarray, is_holiday: np.ndarray) -> float:
    """
    Calculate Weighted Mean Absolute Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        is_holiday: Holiday indicator (1 for holiday, 0 for non-holiday)
    
    Returns:
        WMAE score
    """
    weights = np.where(is_holiday, 5, 1)
    return np.average(np.abs(y_true - y_pred), weights=weights)

# Example usage configuration
def setup_mlflow_logging(dagshub_repo_owner: str, 
                        dagshub_repo_name: str, 
                        dagshub_token: str = None) -> MLflowDagsHubLogger:
    """
    Setup MLflow logging with DagsHub
    
    Args:
        dagshub_repo_owner: Your DagsHub username
        dagshub_repo_name: Repository name
        dagshub_token: Authentication token
    
    Returns:
        Configured logger instance
    """
    logger = MLflowDagsHubLogger(
        dagshub_repo_owner=dagshub_repo_owner,
        dagshub_repo_name=dagshub_repo_name,
        dagshub_token=dagshub_token
    )
    
    return logger