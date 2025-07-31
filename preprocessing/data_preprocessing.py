# data_preprocessing.py
# Data Loading and Preprocessing for Retail Time Series Forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import warnings
warnings.filterwarnings('ignore')

class RetailDataPreprocessor:
    """Data preprocessing class for retail forecasting datasets"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = None
        self.date_column = None
        
    def load_datasets(self, train_path='train.csv', stores_path='stores.csv', features_path='features.csv'):
        """Load all dataset files"""
        
        with mlflow.start_run(run_name="Data_Loading"):
            print("Loading datasets...")
            
            # Load datasets
            train_df = pd.read_csv(train_path)
            stores_df = pd.read_csv(stores_path)
            features_df = pd.read_csv(features_path)
            
            # Log dataset information
            mlflow.log_param("train_shape", train_df.shape)
            mlflow.log_param("stores_shape", stores_df.shape)
            mlflow.log_param("features_shape", features_df.shape)
            
            print(f"Train dataset shape: {train_df.shape}")
            print(f"Stores dataset shape: {stores_df.shape}")
            print(f"Features dataset shape: {features_df.shape}")
            
            # Display dataset info
            print("\n=== TRAIN DATASET ===")
            print("Columns:", train_df.columns.tolist())
            print("Data types:")
            print(train_df.dtypes)
            print("\nFirst 5 rows:")
            print(train_df.head())
            
            print("\n=== STORES DATASET ===")
            print("Columns:", stores_df.columns.tolist())
            print("Data types:")
            print(stores_df.dtypes)
            print("\nFirst 5 rows:")
            print(stores_df.head())
            
            print("\n=== FEATURES DATASET ===")
            print("Columns:", features_df.columns.tolist())
            print("Data types:")
            print(features_df.dtypes)
            print("\nFirst 5 rows:")
            print(features_df.head())
            
            # Log column information
            mlflow.log_param("train_columns", train_df.columns.tolist())
            mlflow.log_param("stores_columns", stores_df.columns.tolist())
            mlflow.log_param("features_columns", features_df.columns.tolist())
            
            return train_df, stores_df, features_df
    
    def identify_columns(self, train_df, stores_df, features_df):
        """Automatically identify key columns in the datasets"""
        
        # Common column name patterns
        date_patterns = ['date', 'Date', 'DATE', 'ds', 'timestamp', 'time']
        target_patterns = ['sales', 'Sales', 'SALES', 'y', 'target', 'weekly_sales', 'WeeklySales']
        store_patterns = ['store', 'Store', 'STORE', 'store_id', 'Store_ID']
        dept_patterns = ['dept', 'Dept', 'DEPT', 'department', 'Department', 'dept_id']
        
        # Identify date column
        self.date_column = None
        for col in train_df.columns:
            if any(pattern in col for pattern in date_patterns):
                self.date_column = col
                break
        
        # Identify target column
        self.target_column = None
        for col in train_df.columns:
            if any(pattern in col for pattern in target_patterns):
                self.target_column = col
                break
        
        # Identify store column
        self.store_column = None
        for col in train_df.columns:
            if any(pattern in col for pattern in store_patterns):
                self.store_column = col
                break
        
        # Identify department column
        self.dept_column = None
        for col in train_df.columns:
            if any(pattern in col for pattern in dept_patterns):
                self.dept_column = col
                break
        
        print(f"Identified columns:")
        print(f"Date column: {self.date_column}")
        print(f"Target column: {self.target_column}")
        print(f"Store column: {self.store_column}")
        print(f"Department column: {self.dept_column}")
        
        return {
            'date': self.date_column,
            'target': self.target_column,
            'store': self.store_column,
            'department': self.dept_column
        }
    
    def merge_datasets(self, train_df, stores_df, features_df):
        """Merge all datasets into a single dataframe"""
        
        with mlflow.start_run(run_name="Data_Merging"):
            print("Merging datasets...")
            
            # Start with train dataset
            merged_df = train_df.copy()
            
            # Merge with stores dataset
            if self.store_column and self.store_column in stores_df.columns:
                merged_df = merged_df.merge(stores_df, on=self.store_column, how='left')
                print(f"Merged with stores dataset. New shape: {merged_df.shape}")
            
            # Merge with features dataset
            merge_cols = []
            if self.store_column and self.store_column in features_df.columns:
                merge_cols.append(self.store_column)
            if self.date_column and self.date_column in features_df.columns:
                merge_cols.append(self.date_column)
            
            if merge_cols:
                merged_df = merged_df.merge(features_df, on=merge_cols, how='left')
                print(f"Merged with features dataset. Final shape: {merged_df.shape}")
            
            # Log merge information
            mlflow.log_param("merged_shape", merged_df.shape)
            mlflow.log_param("merge_columns", merge_cols)
            
            return merged_df
    
    def clean_and_preprocess(self, df):
        """Clean and preprocess the merged dataset"""
        
        with mlflow.start_run(run_name="Data_Cleaning"):
            print("Cleaning and preprocessing data...")
            
            # Make a copy
            clean_df = df.copy()
            
            # Convert date column to datetime
            if self.date_column:
                clean_df[self.date_column] = pd.to_datetime(clean_df[self.date_column])
                
                # Extract date features
                clean_df['year'] = clean_df[self.date_column].dt.year
                clean_df['month'] = clean_df[self.date_column].dt.month
                clean_df['day'] = clean_df[self.date_column].dt.day
                clean_df['dayofweek'] = clean_df[self.date_column].dt.dayofweek
                clean_df['dayofyear'] = clean_df[self.date_column].dt.dayofyear
                clean_df['week'] = clean_df[self.date_column].dt.isocalendar().week
                clean_df['quarter'] = clean_df[self.date_column].dt.quarter
                
                # Add holiday indicators (basic)
                clean_df['is_weekend'] = clean_df['dayofweek'].isin([5, 6]).astype(int)
                clean_df['is_month_start'] = clean_df[self.date_column].dt.is_month_start.astype(int)
                clean_df['is_month_end'] = clean_df[self.date_column].dt.is_month_end.astype(int)
            
            # Handle missing values
            missing_before = clean_df.isnull().sum().sum()
            
            # Fill numerical missing values with median
            numerical_cols = clean_df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if clean_df[col].isnull().sum() > 0:
                    clean_df[col].fillna(clean_df[col].median(), inplace=True)
            
            # Fill categorical missing values with mode
            categorical_cols = clean_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != self.date_column and clean_df[col].isnull().sum() > 0:
                    clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)
            
            missing_after = clean_df.isnull().sum().sum()
            
            # Log cleaning statistics
            mlflow.log_metric("missing_values_before", missing_before)
            mlflow.log_metric("missing_values_after", missing_after)
            
            print(f"Missing values before cleaning: {missing_before}")
            print(f"Missing values after cleaning: {missing_after}")
            
            return clean_df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        
        with mlflow.start_run(run_name="Feature_Encoding"):
            print("Encoding categorical features...")
            
            encoded_df = df.copy()
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Remove date column from categorical encoding
            if self.date_column in categorical_cols:
                categorical_cols = categorical_cols.drop(self.date_column)
            
            for col in categorical_cols:
                if col != self.target_column:  # Don't encode target if it's categorical
                    le = LabelEncoder()
                    encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                    self.label_encoders[col] = le
                    
            # Log encoding information
            mlflow.log_param("encoded_columns", list(categorical_cols))
            mlflow.log_param("num_encoded_features", len(categorical_cols))
            
            print(f"Encoded {len(categorical_cols)} categorical columns")
            
            return encoded_df
    
    def create_time_series_features(self, df):
        """Create time series specific features"""
        
        with mlflow.start_run(run_name="Time_Series_Feature_Creation"):
            print("Creating time series features...")
            
            ts_df = df.copy()
            
            # Sort by date and store/department if available
            sort_cols = [self.date_column]
            if self.store_column:
                sort_cols.append(self.store_column)
            if self.dept_column:
                sort_cols.append(self.dept_column)
            
            ts_df = ts_df.sort_values(sort_cols)
            
            # Create lag features
            if self.target_column:
                # Group by store and department if available
                group_cols = []
                if self.store_column:
                    group_cols.append(self.store_column)
                if self.dept_column:
                    group_cols.append(self.dept_column)
                
                if group_cols:
                    # Create lag features for each group
                    for lag in [1, 2, 3, 4, 7, 14, 28]:
                        ts_df[f'{self.target_column}_lag_{lag}'] = (
                            ts_df.groupby(group_cols)[self.target_column].shift(lag)
                        )
                    
                    # Create rolling statistics
                    for window in [7, 14, 28]:
                        ts_df[f'{self.target_column}_rolling_mean_{window}'] = (
                            ts_df.groupby(group_cols)[self.target_column]
                            .rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
                        )
                        ts_df[f'{self.target_column}_rolling_std_{window}'] = (
                            ts_df.groupby(group_cols)[self.target_column]
                            .rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
                        )
                else:
                    # Create lag features without grouping
                    for lag in [1, 2, 3, 4, 7, 14, 28]:
                        ts_df[f'{self.target_column}_lag_{lag}'] = ts_df[self.target_column].shift(lag)
                    
                    # Create rolling statistics
                    for window in [7, 14, 28]:
                        ts_df[f'{self.target_column}_rolling_mean_{window}'] = (
                            ts_df[self.target_column].rolling(window=window, min_periods=1).mean()
                        )
                        ts_df[f'{self.target_column}_rolling_std_{window}'] = (
                            ts_df[self.target_column].rolling(window=window, min_periods=1).std()
                        )
            
            # Log feature creation
            new_features = [col for col in ts_df.columns if col not in df.columns]
            mlflow.log_param("new_time_series_features", new_features)
            mlflow.log_param("num_new_features", len(new_features))
            
            print(f"Created {len(new_features)} new time series features")
            
            return ts_df
    
    def prepare_for_modeling(self, df, target_col=None, test_size=0.2):
        """Prepare data for modeling"""
        
        with mlflow.start_run(run_name="Model_Data_Preparation"):
            print("Preparing data for modeling...")
            
            # Use identified target column if not specified
            if target_col is None:
                target_col = self.target_column
            
            # Remove non-feature columns
            exclude_cols = [self.date_column, target_col]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Handle any remaining missing values in features
            df_modeling = df[feature_cols + [target_col, self.date_column]].copy()
            df_modeling = df_modeling.dropna()
            
            # Sort by date
            df_modeling = df_modeling.sort_values(self.date_column)
            
            # Split data temporally
            split_point = int(len(df_modeling) * (1 - test_size))
            
            train_data = df_modeling.iloc[:split_point]
            test_data = df_modeling.iloc[split_point:]
            
            # Log preparation statistics
            mlflow.log_param("feature_columns", feature_cols)
            mlflow.log_param("target_column", target_col)
            mlflow.log_param("train_size", len(train_data))
            mlflow.log_param("test_size", len(test_data))
            mlflow.log_param("total_features", len(feature_cols))
            
            print(f"Training data: {len(train_data)} samples")
            print(f"Test data: {len(test_data)} samples")
            print(f"Number of features: {len(feature_cols)}")
            
            return train_data, test_data, feature_cols
    
    def create_prophet_format(self, df, target_col=None):
        """Convert data to Prophet format (ds, y columns)"""
        
        # Use identified columns if not specified
        if target_col is None:
            target_col = self.target_column
        
        # Create Prophet format dataframe
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df[self.date_column]),
            'y': df[target_col]
        })
        
        # Add additional regressors (numerical columns only)
        exclude_cols = [self.date_column, target_col, 'ds', 'y']
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        regressor_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        for col in regressor_cols:
            prophet_df[col] = df[col]
        
        return prophet_df, regressor_cols
    
    def plot_data_overview(self, df):
        """Create comprehensive data overview plots"""
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # Time series plot
        if self.date_column and self.target_column:
            ts_data = df.groupby(self.date_column)[self.target_column].sum()
            axes[0, 0].plot(ts_data.index, ts_data.values)
            axes[0, 0].set_title(f'Total {self.target_column} Over Time')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Distribution of target
        if self.target_column:
            axes[0, 1].hist(df[self.target_column], bins=50, alpha=0.7)
            axes[0, 1].set_title(f'Distribution of {self.target_column}')
        
        # Store-wise sales (if store column exists)
        if self.store_column and self.target_column:
            store_sales = df.groupby(self.store_column)[self.target_column].sum().sort_values(ascending=False)
            axes[1, 0].bar(range(len(store_sales)), store_sales.values)
            axes[1, 0].set_title('Sales by Store')
            axes[1, 0].set_xlabel('Store (ranked)')
        
        # Department-wise sales (if dept column exists)
        if self.dept_column and self.target_column:
            dept_sales = df.groupby(self.dept_column)[self.target_column].sum().sort_values(ascending=False)
            axes[1, 1].bar(range(len(dept_sales)), dept_sales.values)
            axes[1, 1].set_title('Sales by Department')
            axes[1, 1].set_xlabel('Department (ranked)')
        
        # Monthly seasonality
        if self.date_column and self.target_column:
            df_temp = df.copy()
            df_temp['month'] = pd.to_datetime(df_temp[self.date_column]).dt.month
            monthly_sales = df_temp.groupby('month')[self.target_column].mean()
            axes[2, 0].plot(monthly_sales.index, monthly_sales.values, marker='o')
            axes[2, 0].set_title('Average Sales by Month')
            axes[2, 0].set_xlabel('Month')
        
        # Weekly seasonality
        if self.date_column and self.target_column:
            df_temp = df.copy()
            df_temp['dayofweek'] = pd.to_datetime(df_temp[self.date_column]).dt.dayofweek
            weekly_sales = df_temp.groupby('dayofweek')[self.target_column].mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            axes[2, 1].plot(weekly_sales.index, weekly_sales.values, marker='o')
            axes[2, 1].set_title('Average Sales by Day of Week')
            axes[2, 1].set_xticks(range(7))
            axes[2, 1].set_xticklabels(days)
        
        plt.tight_layout()
        plt.savefig('data_overview_complete.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('data_overview_complete.png')
        plt.show()
    
    def run_complete_preprocessing(self, train_path='train.csv', stores_path='stores.csv', 
                                 features_path='features.csv'):
        """Run the complete preprocessing pipeline"""
        
        # Set MLflow experiment
        mlflow.set_experiment("Data_Preprocessing")
        
        # Load datasets
        train_df, stores_df, features_df = self.load_datasets(train_path, stores_path, features_path)
        
        # Identify columns
        column_mapping = self.identify_columns(train_df, stores_df, features_df)
        
        # Merge datasets
        merged_df = self.merge_datasets(train_df, stores_df, features_df)
        
        # Clean and preprocess
        clean_df = self.clean_and_preprocess(merged_df)
        
        # Encode categorical features
        encoded_df = self.encode_categorical_features(clean_df)
        
        # Create time series features
        ts_df = self.create_time_series_features(encoded_df)
        
        # Plot data overview
        self.plot_data_overview(ts_df)
        
        # Prepare for modeling
        train_data, test_data, feature_cols = self.prepare_for_modeling(ts_df)
        
        # Create Prophet format
        prophet_train, regressor_cols = self.create_prophet_format(train_data)
        prophet_test, _ = self.create_prophet_format(test_data)
        
        return {
            'train_data': train_data,
            'test_data': test_data,
            'feature_columns': feature_cols,
            'prophet_train': prophet_train,
            'prophet_test': prophet_test,
            'regressor_columns': regressor_cols,
            'column_mapping': column_mapping,
            'preprocessor': self
        }

# Example usage
def run_preprocessing_pipeline():
    """Run the complete preprocessing pipeline"""
    
    preprocessor = RetailDataPreprocessor()
    
    # Run preprocessing
    results = preprocessor.run_complete_preprocessing(
        train_path='train.csv',
        stores_path='stores.csv', 
        features_path='features.csv'
    )
    
    print("Preprocessing completed!")
    print(f"Column mapping: {results['column_mapping']}")
    print(f"Feature columns: {len(results['feature_columns'])}")
    print(f"Regressor columns for Prophet: {len(results['regressor_columns'])}")
    
    return results

if __name__ == "__main__":
    results = run_preprocessing_pipeline()
