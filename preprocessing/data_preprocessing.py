# src/data_preprocessor.py

import pandas as pd
from typing import Dict, Tuple, Optional, List
from . import config


class WalmartDataPreprocessor:
    """
    A comprehensive preprocessor for Walmart recruiting dataset.
    Handles merging, transformation, and aggregation operations.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.processed_cache = {}
    
    def _combine_with_features(
        self, 
        primary_df: pd.DataFrame, 
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge primary dataset with features data.
        
        Args:
            primary_df: Main dataset (train/test)
            features_df: Features dataset
            
        Returns:
            Merged DataFrame
        """
        merge_keys = ['Store', 'Date', 'IsHoliday']
        return pd.merge(primary_df, features_df, on=merge_keys, how='left')
    
    def _combine_with_stores(
        self, 
        primary_df: pd.DataFrame, 
        stores_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge primary dataset with stores metadata.
        
        Args:
            primary_df: Main dataset
            stores_df: Stores metadata
            
        Returns:
            Merged DataFrame
        """
        return pd.merge(primary_df, stores_df, on=['Store'], how='left')
    
    def _handle_dates_and_ordering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process date columns and sort the dataframe.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with processed dates and sorted rows
        """
        if config.DATE_COLUMN in df.columns:
            df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
            
            # Determine sorting columns based on what's available
            ordering_cols = []
            for col in [config.DATE_COLUMN, 'Store', 'Dept']:
                if col in df.columns:
                    ordering_cols.append(col)
            
            if ordering_cols:
                df = df.sort_values(by=ordering_cols).reset_index(drop=True)
        
        return df
    
    def transform_datasets(
        self,
        data_collection: Dict[str, pd.DataFrame],
        include_train: bool = True,
        include_test: bool = True,
        attach_features: bool = True,
        attach_stores: bool = True,
        remove_source_data: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply comprehensive preprocessing pipeline to selected datasets.
        
        Args:
            data_collection: Dictionary of raw DataFrames
            include_train: Whether to process training data
            include_test: Whether to process test data
            attach_features: Whether to merge features data
            attach_stores: Whether to merge stores data
            remove_source_data: Whether to delete source data after processing
            
        Returns:
            Dictionary of processed DataFrames
        """
        datasets_to_transform = []
        
        if include_train and "train" in data_collection:
            datasets_to_transform.append("train")
        if include_test and "test" in data_collection:
            datasets_to_transform.append("test")
        
        if not datasets_to_transform:
            print("Warning: No datasets selected for transformation.")
            return {}
        
        transformed_data = {}
        
        for dataset_name in datasets_to_transform:
            df = data_collection[dataset_name].copy()
            
            # Apply feature merging if requested
            if attach_features and "features" in data_collection:
                df = self._combine_with_features(df, data_collection["features"])
            
            # Apply stores merging if requested
            if attach_stores and "stores" in data_collection:
                df = self._combine_with_stores(df, data_collection["stores"])
            
            # Process dates and sort
            df = self._handle_dates_and_ordering(df)
            
            transformed_data[dataset_name] = df
        
        # Clean up source data if requested
        if remove_source_data:
            cleanup_keys = datasets_to_transform.copy()
            if attach_features:
                cleanup_keys.append("features")
            if attach_stores:
                cleanup_keys.append("stores")
            
            for key in cleanup_keys:
                if key in data_collection:
                    del data_collection[key]
        
        return transformed_data
    
    def aggregate_to_store_level(
        self,
        data_collection: Dict[str, pd.DataFrame],
        include_train: bool = True,
        include_test: bool = True,
        cleanup_sources: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Aggregate department-level data to store level.
        
        Args:
            data_collection: Dictionary of raw DataFrames
            include_train: Whether to process training data
            include_test: Whether to process test data
            cleanup_sources: Whether to remove source data after processing
            
        Returns:
            Dictionary of store-level aggregated DataFrames
        """
        datasets_to_aggregate = []
        
        # Determine which datasets to process
        if include_train and 'train' in data_collection:
            datasets_to_aggregate.append('train')
        if include_test and 'test' in data_collection:
            datasets_to_aggregate.append('test')
        
        if not datasets_to_aggregate:
            print("Warning: No datasets available for store-level aggregation.")
            return {}
        
        # Verify required auxiliary data exists
        required_data = ['features', 'stores']
        for req in required_data:
            if req not in data_collection:
                raise KeyError(f"Required dataset '{req}' missing from data collection")
        
        store_aggregated = {}
        features_data = data_collection['features']
        stores_data = data_collection['stores']
        
        for dataset_name in datasets_to_aggregate:
            source_df = data_collection[dataset_name]
            aggregation_keys = ['Store', config.DATE_COLUMN, 'IsHoliday']
            
            if dataset_name == 'train':
                # For training data, aggregate sales by taking mean
                aggregated_df = (source_df.groupby(aggregation_keys)
                               [config.TARGET_COLUMN].mean().reset_index())
            else:
                # For test data, just get unique combinations
                aggregated_df = (source_df[aggregation_keys]
                               .drop_duplicates().reset_index(drop=True))
            
            # Merge with auxiliary data
            aggregated_df = self._combine_with_stores(aggregated_df, stores_data)
            aggregated_df = self._combine_with_features(aggregated_df, features_data)
            aggregated_df = self._handle_dates_and_ordering(aggregated_df)
            
            store_aggregated[dataset_name] = aggregated_df
        
        # Cleanup if requested
        if cleanup_sources:
            cleanup_targets = ['features', 'stores'] + datasets_to_aggregate
            for target in set(cleanup_targets):
                if target in data_collection:
                    del data_collection[target]
        
        return store_aggregated
    
    def create_temporal_split(
        self,
        dataset: pd.DataFrame,
        split_target: bool = True,
        target_col: str = config.TARGET_COLUMN
    ) -> Tuple:
        """
        Split data based on configured date threshold.
        
        Args:
            dataset: DataFrame to split
            split_target: Whether to separate target variable
            target_col: Name of target column
            
        Returns:
            Tuple of split data (X_train, y_train, X_val, y_val) or (train_df, val_df)
        """
        train_portion = dataset[dataset["Date"] < config.SPLIT_DATE]
        validation_portion = dataset[dataset["Date"] >= config.SPLIT_DATE]
        
        if split_target:
            X_train = train_portion.drop(columns=[target_col])
            y_train = train_portion[target_col]
            X_val = validation_portion.drop(columns=[target_col])
            y_val = validation_portion[target_col]
            return X_train, y_train, X_val, y_val
        
        return train_portion, validation_portion
    
    def create_ratio_split(
        self,
        dataset: pd.DataFrame,
        split_target: bool = True,
        target_col: str = config.TARGET_COLUMN
    ) -> Tuple:
        """
        Split data based on configured ratio.
        
        Args:
            dataset: DataFrame to split
            split_target: Whether to separate target variable
            target_col: Name of target column
            
        Returns:
            Tuple of split data (X_train, y_train, X_val, y_val) or (train_df, val_df)
        """
        cutoff_idx = int(config.TRAIN_RATIO * len(dataset))
        train_portion = dataset.iloc[:cutoff_idx]
        validation_portion = dataset.iloc[cutoff_idx:]
        
        if split_target:
            X_train = train_portion.drop(columns=[target_col])
            y_train = train_portion[target_col]
            X_val = validation_portion.drop(columns=[target_col])
            y_val = validation_portion[target_col]
            return X_train, y_train, X_val, y_val
        
        return train_portion, validation_portion
    
    def get_preprocessing_summary(
        self, 
        data_before: Dict[str, pd.DataFrame],
        data_after: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Generate a summary of preprocessing changes.
        
        Args:
            data_before: Data before preprocessing
            data_after: Data after preprocessing
            
        Returns:
            Summary dictionary with before/after statistics
        """
        summary = {
            "before": {},
            "after": {},
            "changes": {}
        }
        
        for name in data_before.keys():
            summary["before"][name] = {
                "shape": data_before[name].shape,
                "columns": list(data_before[name].columns)
            }
        
        for name in data_after.keys():
            summary["after"][name] = {
                "shape": data_after[name].shape,
                "columns": list(data_after[name].columns)
            }
            
            if name in data_before:
                summary["changes"][name] = {
                    "shape_change": (data_after[name].shape[0] - data_before[name].shape[0],
                                   data_after[name].shape[1] - data_before[name].shape[1]),
                    "new_columns": set(data_after[name].columns) - set(data_before[name].columns),
                    "removed_columns": set(data_before[name].columns) - set(data_after[name].columns)
                }
        
        return summary


# Convenience functions
def preprocess_walmart_data(
    raw_data: Dict[str, pd.DataFrame],
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function for basic preprocessing.
    
    Args:
        raw_data: Dictionary of raw DataFrames
        **kwargs: Additional arguments for transform_datasets
        
    Returns:
        Dictionary of processed DataFrames
    """
    preprocessor = WalmartDataPreprocessor()
    return preprocessor.transform_datasets(raw_data, **kwargs)