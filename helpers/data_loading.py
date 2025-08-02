# src/data_loader.py

import pandas as pd
from typing import Optional, List, Dict, Union
from pathlib import Path
from . import config


class WalmartDataLoader:
    """
    A data loader class for handling Walmart recruiting dataset files.
    Provides methods to load individual files or multiple files at once.
    """
    
    def __init__(self):
        """Initialize the data loader with available dataset mappings."""
        self.dataset_registry = {
            "stores": config.STORES_PATH,
            "features": config.FEATURES_PATH,
            "train": config.TRAIN_PATH,
            "test": config.TEST_PATH,
            "sample_submission": config.SAMPLE_SUBMISSION_PATH
        }
    
    def get_available_datasets(self) -> List[str]:
        """
        Returns a list of all available dataset names.
        
        Returns:
            List[str]: Available dataset identifiers.
        """
        return list(self.dataset_registry.keys())
    
    def load_single_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load a single dataset by name.
        
        Args:
            dataset_name (str): Name of the dataset to load.
            
        Returns:
            pd.DataFrame: The loaded dataset.
            
        Raises:
            ValueError: If dataset name is not recognized.
            FileNotFoundError: If the file path doesn't exist.
        """
        if dataset_name not in self.dataset_registry:
            available = ", ".join(self.dataset_registry.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available options: {available}"
            )
        
        file_path = self.dataset_registry[dataset_name]
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found at path: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded '{dataset_name}' with shape {df.shape}")
            return df
        except Exception as e:
            raise Exception(f"Error loading '{dataset_name}': {str(e)}")
    
    def load_multiple_datasets(
        self, 
        dataset_names: Optional[List[str]] = None,
        skip_missing: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple datasets at once.
        
        Args:
            dataset_names (Optional[List[str]]): List of dataset names to load.
                If None, loads all available datasets.
            skip_missing (bool): If True, skips datasets that can't be loaded
                instead of raising an error.
                
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping dataset names to DataFrames.
        """
        if dataset_names is None:
            dataset_names = self.get_available_datasets()
        
        loaded_data = {}
        errors = []
        
        for name in dataset_names:
            try:
                loaded_data[name] = self.load_single_dataset(name)
            except Exception as e:
                error_msg = f"Failed to load '{name}': {str(e)}"
                if skip_missing:
                    print(f"Warning: {error_msg}")
                    continue
                else:
                    errors.append(error_msg)
        
        if errors and not skip_missing:
            raise Exception("Loading failed:\n" + "\n".join(errors))
        
        print(f"Data loading completed. Loaded {len(loaded_data)} datasets.")
        return loaded_data
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Union[str, int, tuple]]:
        """
        Get basic information about a dataset without fully loading it.
        
        Args:
            dataset_name (str): Name of the dataset.
            
        Returns:
            Dict containing dataset path, shape, and columns info.
        """
        if dataset_name not in self.dataset_registry:
            raise ValueError(f"Dataset '{dataset_name}' not recognized.")
        
        file_path = self.dataset_registry[dataset_name]
        
        if not Path(file_path).exists():
            return {
                "path": file_path,
                "exists": False,
                "error": "File not found"
            }
        
        try:
            # Read just the first few rows to get structure info
            sample_df = pd.read_csv(file_path, nrows=5)
            full_df = pd.read_csv(file_path)
            
            return {
                "path": file_path,
                "exists": True,
                "shape": full_df.shape,
                "columns": list(sample_df.columns),
                "dtypes": dict(sample_df.dtypes),
                "sample_data": sample_df.head(3).to_dict()
            }
        except Exception as e:
            return {
                "path": file_path,
                "exists": True,
                "error": f"Error reading file: {str(e)}"
            }


# Convenience functions for backward compatibility
def fetch_walmart_data(
    datasets_to_fetch: Optional[List[str]] = None,
    skip_errors: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load Walmart datasets.
    
    Args:
        datasets_to_fetch (Optional[List[str]]): List of dataset names to load.
            If None, loads all available datasets.
        skip_errors (bool): Whether to skip datasets that fail to load.
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of loaded datasets.
    """
    loader = WalmartDataLoader()
    return loader.load_multiple_datasets(datasets_to_fetch, skip_errors)


def get_dataset_summary() -> Dict[str, Dict]:
    """
    Get summary information for all available datasets.
    
    Returns:
        Dict containing information about each dataset.
    """
    loader = WalmartDataLoader()
    summary = {}
    
    for dataset_name in loader.get_available_datasets():
        summary[dataset_name] = loader.get_dataset_info(dataset_name)
    
    return summary