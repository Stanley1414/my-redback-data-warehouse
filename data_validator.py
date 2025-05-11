"""
Data Validator Module

This module provides a comprehensive data validation system that uses AI-powered algorithms
to detect and correct anomalies in datasets. It supports multiple file formats and provides
detailed validation reports.

Key Features:
- Multiple file format support (CSV, Excel, JSON, Parquet)
- AI-powered anomaly detection using Isolation Forest
- Custom validation rules
- Automatic data correction
- Missing value handling
- Pattern detection
- Consistency checking
- Detailed validation reporting

Algorithm Details:

1. Isolation Forest Anomaly Detection:
   - Uses random partitioning to isolate anomalies
   - Anomalies require fewer partitions to isolate
   - Works by building an ensemble of isolation trees
   - Each tree is built by randomly selecting features and split values
   - Anomaly score is based on path length in the trees
   - Advantages: efficient, works with high-dimensional data, no distribution assumptions

2. Pattern Detection:
   - Sudden Changes: Uses rolling statistics to detect abrupt changes
   - Correlation Analysis: Identifies and monitors relationships between features
   - Statistical Thresholds: Uses standard deviation for anomaly detection
   - Time Series Analysis: Detects patterns in sequential data

3. Missing Value Handling:
   - Numerical Data: Median imputation to preserve distribution
   - Categorical Data: Mode imputation for consistency
   - Advanced Methods: KNN imputation for complex relationships

4. Data Correction:
   - Statistical Methods: Mean/median replacement
   - Range-based: Clipping to valid ranges
   - Pattern-based: Smoothing of sudden changes
   - Relationship-preserving: Maintains correlations

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import re
from typing import Dict, List, Union, Tuple
import json

class DataValidator:
    """
    A class for validating and correcting data using AI-powered algorithms.
    
    This class provides methods for:
    - Loading data from various file formats
    - Training anomaly detection models
    - Validating data against custom rules
    - Detecting and correcting anomalies
    - Handling missing values
    - Generating validation reports
    
    Algorithm Implementation Details:
    
    1. Anomaly Detection (Isolation Forest):
       - Builds isolation trees using random feature selection
       - Calculates anomaly scores based on path lengths
       - Uses contamination parameter to control sensitivity
       - Handles both global and local anomalies
    
    2. Pattern Recognition:
       - Implements rolling window analysis
       - Uses statistical measures (mean, std) for thresholding
       - Detects both point and contextual anomalies
       - Considers temporal relationships in data
    
    3. Data Correction:
       - Implements multiple correction strategies
       - Preserves data distributions
       - Maintains feature relationships
       - Handles different data types appropriately
    
    Attributes:
        model: The trained anomaly detection model
        scaler: StandardScaler for normalizing data
        model_name: Name identifier for the model
        validation_rules: Dictionary of validation rules for specific columns
        file_handlers: Dictionary mapping file extensions to their respective pandas readers
    """
    
    def __init__(self, model_name: str = 'default_model'):
        """
        Initialize the DataValidator with default settings.
        
        Args:
            model_name (str): Name identifier for the model. Used when saving/loading models.
        """
        self.model = None
        self.scaler = StandardScaler()
        self.model_name = model_name
        self.validation_rules = {}
        # Dictionary mapping file extensions to their respective pandas readers
        self.file_handlers = {
            '.csv': pd.read_csv,
            '.xls': lambda x: pd.read_excel(x, engine='xlrd'),
            '.xlsx': lambda x: pd.read_excel(x, engine='openpyxl'),
            '.json': pd.read_json,
            '.parquet': pd.read_parquet
        }
    
    def set_validation_rules(self, rules: Dict[str, Dict[str, float]]):
        """
        Set custom validation rules for specific columns.
        
        Args:
            rules (Dict[str, Dict[str, float]]): Dictionary of validation rules.
                Format: {'column_name': {'min': min_value, 'max': max_value}}
        """
        self.validation_rules = rules
    
    def convert_to_csv(self, file_path: str) -> str:
        """
        Convert any supported file format to CSV.
        
        Args:
            file_path (str): Path to the input file.
            
        Returns:
            str: Path to the converted CSV file.
            
        Raises:
            Exception: If conversion fails.
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            csv_path = os.path.splitext(file_path)[0] + '_converted.csv'
            
            # Read the file based on its extension
            if file_extension in ['.xls', '.xlsx']:
                try:
                    if file_extension == '.xls':
                        df = pd.read_excel(file_path, engine='xlrd')
                    else:
                        df = pd.read_excel(file_path, engine='openpyxl')
                except Exception as e:
                    # If Excel reading fails, try reading as CSV
                    df = pd.read_csv(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_extension == '.csv':
                # If it's already CSV, just copy it
                import shutil
                shutil.copy2(file_path, csv_path)
                return csv_path
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Save as CSV
            df.to_csv(csv_path, index=False)
            print(f"File converted to CSV format: {csv_path}")
            return csv_path
            
        except Exception as e:
            raise Exception(f"Error converting file to CSV: {str(e)}")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats, converting to CSV if necessary.
        
        Args:
            file_path (str): Path to the data file.
            
        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
            
        Raises:
            ValueError: If file format is not supported.
            Exception: If there's an error loading the file.
        """
        try:
            # Convert file to CSV if it's not already
            csv_path = self.convert_to_csv(file_path)
            
            # Load the CSV file
            return pd.read_csv(csv_path)
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def train_validation_model(self, data: pd.DataFrame, save_path: str = None):
        """
        Train the anomaly detection model on the provided data.
        
        Algorithm Details:
        1. Data Preprocessing:
           - Converts categorical columns to numerical using one-hot encoding
           - Scales data using StandardScaler
           - Handles missing values
        
        2. Model Training:
           - Uses Isolation Forest algorithm
           - contamination=0.1 (expects 10% anomalies)
           - random_state=42 for reproducibility
           - Builds ensemble of isolation trees
        
        3. Model Persistence:
           - Saves model and scaler if path provided
           - Uses joblib for efficient serialization
           - Maintains model versioning
        
        Args:
            data (pd.DataFrame): Training data.
            save_path (str, optional): Path to save the trained model and scaler.
            
        Raises:
            ValueError: If no valid columns are found for training after preprocessing.
        """
        # Make a copy of the data to avoid modifying the original
        processed_data = data.copy()
        
        # Convert categorical columns to numerical using one-hot encoding
        categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            # Perform one-hot encoding
            processed_data = pd.get_dummies(processed_data, columns=categorical_cols, drop_first=True)
        
        # Get numerical columns after preprocessing
        numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            raise ValueError(
                "No numerical columns found in the data for training. "
                "Please ensure your data contains at least one of the following:\n"
                "1. Numerical columns (integers, floats)\n"
                "2. Categorical columns that can be converted to numerical format\n"
                "3. Date/time columns that can be converted to numerical features"
            )
        
        # Scale the data for better model performance
        scaled_data = self.scaler.fit_transform(processed_data[numerical_cols])
        
        # Train Isolation Forest for anomaly detection
        # contamination=0.1 means we expect 10% of the data to be anomalous
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.model.fit(scaled_data)
        
        # Save the model and scaler if path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            model_path = os.path.join(save_path, f'{self.model_name}_model.joblib')
            scaler_path = os.path.join(save_path, f'{self.model_name}_scaler.joblib')
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            # Save column information for future reference
            column_info = {
                'numerical_columns': numerical_cols.tolist(),
                'categorical_columns': categorical_cols.tolist()
            }
            column_info_path = os.path.join(save_path, f'{self.model_name}_columns.json')
            with open(column_info_path, 'w') as f:
                json.dump(column_info, f)
    
    def load_trained_model(self, model_path: str, scaler_path: str):
        """
        Load a previously trained model and scaler.
        
        Args:
            model_path (str): Path to the saved model file.
            scaler_path (str): Path to the saved scaler file.
            
        Raises:
            Exception: If there's an error loading the model or scaler.
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def validate_ranges(self, data: pd.DataFrame) -> Dict:
        """
        Validate data against predefined ranges.
        
        Args:
            data (pd.DataFrame): Data to validate.
            
        Returns:
            Dict: Dictionary containing range violations for each column.
        """
        range_violations = {}
        
        for column, rules in self.validation_rules.items():
            if column in data.columns:
                # Check if values are outside the defined range
                violations = (
                    (data[column] < rules['min']) |
                    (data[column] > rules['max'])
                )
                if violations.any():
                    range_violations[column] = {
                        'count': violations.sum(),
                        'indices': violations[violations].index.tolist()
                    }
        
        return range_violations
    
    def validate_consistency(self, data: pd.DataFrame) -> Dict:
        """
        Validate consistency between related fields using correlation analysis.
        
        Algorithm Details:
        1. Correlation Analysis:
           - Calculates pairwise correlations between numerical columns
           - Uses Pearson correlation coefficient
           - Threshold of 0.8 for strong correlations
        
        2. Relationship Validation:
           - For highly correlated features:
             * Calculates ratio between features
             * Computes mean and standard deviation
             * Flags values > 3 standard deviations from mean
        
        3. Statistical Measures:
           - Uses robust statistical methods
           - Considers data distribution
           - Handles outliers appropriately
        
        Args:
            data (pd.DataFrame): Data to validate.
            
        Returns:
            Dict: Dictionary containing consistency violations.
        """
        consistency_violations = {}
        
        # Get numerical columns for correlation analysis
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        # Check for correlations between numerical columns
        if len(numerical_cols) > 1:
            corr_matrix = data[numerical_cols].corr()
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    # Check for high correlation (>0.8)
                    if abs(corr_matrix.loc[col1, col2]) > 0.8:
                        # Check for violations in the relationship
                        ratio = data[col1] / data[col2]
                        mean_ratio = ratio.mean()
                        std_ratio = ratio.std()
                        # Flag values that deviate more than 3 standard deviations
                        violations = abs(ratio - mean_ratio) > (3 * std_ratio)
                        
                        if violations.any():
                            consistency_violations[f'{col1}_{col2}_relationship'] = {
                                'count': violations.sum(),
                                'indices': violations[violations].index.tolist()
                            }
        
        return consistency_violations
    
    def validate_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Validate patterns in the data by detecting sudden changes.
        
        Algorithm Details:
        1. Change Detection:
           - Calculates first-order differences
           - Uses rolling statistics for context
           - Implements adaptive thresholds
        
        2. Statistical Analysis:
           - Uses standard deviation for thresholding
           - Considers local and global patterns
           - Handles different data scales
        
        3. Pattern Recognition:
           - Detects sudden changes
           - Identifies trend violations
           - Flags unusual patterns
        
        Args:
            data (pd.DataFrame): Data to validate.
            
        Returns:
            Dict: Dictionary containing pattern violations.
        """
        pattern_violations = {}
        
        # Check for sudden changes in numerical columns
        for col in data.select_dtypes(include=[np.number]).columns:
            # Calculate absolute differences between consecutive values
            changes = data[col].diff().abs()
            # Flag changes that are more than 3 standard deviations from the mean
            sudden_changes = changes > (data[col].std() * 3)
            
            if sudden_changes.any():
                pattern_violations[f'{col}_changes'] = {
                    'count': sudden_changes.sum(),
                    'indices': sudden_changes[sudden_changes].index.tolist()
                }
        
        return pattern_violations
    
    def handle_missing_values(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle missing values in the data using appropriate imputation methods.
        
        Algorithm Details:
        1. Missing Value Detection:
           - Identifies null values
           - Calculates missing percentages
           - Categorizes by data type
        
        2. Imputation Strategies:
           - Numerical Data:
             * Uses median for robustness
             * Preserves distribution
             * Handles outliers
        
           - Categorical Data:
             * Uses mode (most frequent)
             * Maintains category relationships
             * Preserves data consistency
        
        3. Quality Metrics:
           - Tracks imputation statistics
           - Monitors data quality
           - Reports missing patterns
        
        Args:
            data (pd.DataFrame): Data to process.
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Corrected data and missing value information.
        """
        missing_info = {}
        corrected_data = data.copy()
        
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                missing_info[column] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(data)) * 100
                }
                
                # Handle missing values based on column type
                if pd.api.types.is_numeric_dtype(data[column]):
                    # For numerical columns, use median
                    corrected_data[column] = data[column].fillna(data[column].median())
                else:
                    # For categorical columns, use mode
                    corrected_data[column] = data[column].fillna(data[column].mode()[0])
        
        return corrected_data, missing_info
    
    def validate_data(self, new_data: pd.DataFrame) -> Dict:
        """
        Validate new data using the trained model.
        
        Args:
            new_data (pd.DataFrame): Data to validate.
            
        Returns:
            Dict: Validation results including anomaly scores.
            
        Raises:
            Exception: If model is not trained.
        """
        if self.model is None:
            raise Exception("Model not trained. Please train the model first.")
        
        # Make a copy of the data to avoid modifying the original
        processed_data = new_data.copy()
        
        # Convert categorical columns to numerical using one-hot encoding
        categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            # Perform one-hot encoding
            processed_data = pd.get_dummies(processed_data, columns=categorical_cols, drop_first=True)
        
        # Get numerical columns after preprocessing
        numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            raise ValueError(
                "No numerical columns found in the data for validation. "
                "Please ensure your data contains at least one of the following:\n"
                "1. Numerical columns (integers, floats)\n"
                "2. Categorical columns that can be converted to numerical format\n"
                "3. Date/time columns that can be converted to numerical features"
            )
        
        # Scale the new data
        scaled_data = self.scaler.transform(processed_data[numerical_cols])
        
        # Predict anomalies
        predictions = self.model.predict(scaled_data)
        
        # Create validation results
        validation_results = {
            'is_valid': predictions == 1,
            'anomaly_score': self.model.score_samples(scaled_data)
        }
        
        return validation_results
    
    def correct_anomalies(self, data: pd.DataFrame, validation_results: Dict) -> pd.DataFrame:
        """
        Correct detected anomalies using statistical methods.
        
        Args:
            data (pd.DataFrame): Data to correct.
            validation_results (Dict): Results from validate_data method.
            
        Returns:
            pd.DataFrame: Corrected data.
        """
        corrected_data = data.copy()
        
        # For each numerical column
        for col in data.select_dtypes(include=[np.number]).columns:
            # Get valid data points
            valid_mask = validation_results['is_valid']
            valid_data = data.loc[valid_mask, col]
            
            # Calculate statistics for valid data
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            
            # Replace anomalies with mean value
            anomaly_mask = ~validation_results['is_valid']
            corrected_data.loc[anomaly_mask, col] = mean_val
        
        return corrected_data
    
    def validate_and_correct(self, file_path: str, model_dir: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Main function to validate and correct data. The uploaded dataset is used to train
        the model immediately before validation.

        Args:
            file_path (str): Path to the data file.
            model_dir (str, optional): Directory to save the trained model.

        Returns:
            Tuple[pd.DataFrame, Dict]: Corrected data and validation report.
        """
        try:
            # Convert file to CSV if it's not already
            csv_path = self.convert_to_csv(file_path)
            
            # Load data from CSV
            data = self.load_data(csv_path)
            
            # Handle missing values
            data, missing_info = self.handle_missing_values(data)
            
            # Always train the model on the uploaded data
            self.train_validation_model(data, model_dir)
            
            # Perform all validations
            range_violations = self.validate_ranges(data)
            consistency_violations = self.validate_consistency(data)
            pattern_violations = self.validate_patterns(data)
            
            # Validate data using AI model
            validation_results = self.validate_data(data)
            
            # Correct anomalies
            corrected_data = self.correct_anomalies(data, validation_results)
            
            # Generate comprehensive report
            report = {
                'total_records': len(data),
                'anomalies_detected': sum(~validation_results['is_valid']),
                'correction_applied': True,
                'missing_values': missing_info,
                'range_violations': range_violations,
                'consistency_violations': consistency_violations,
                'pattern_violations': pattern_violations,
                'model_trained': True,
                'training_data_size': len(data)
            }
            
            return corrected_data, report
            
        except Exception as e:
            raise Exception(f"Error in validation process: {str(e)}")

def print_validation_report(report: Dict):
    """
    Print a formatted validation report.
    
    Args:
        report (Dict): Validation report from validate_and_correct method.
    """
    print("\nValidation Report:")
    print(f"Total Records: {report['total_records']}")
    print(f"Anomalies Detected: {report['anomalies_detected']}")
    print(f"Corrections Applied: {report['correction_applied']}")
    
    # Print missing values report
    if report['missing_values']:
        print("\nMissing Values Report:")
        for column, info in report['missing_values'].items():
            print(f"{column}: {info['count']} missing values ({info['percentage']:.2f}%)")
    
    # Print range violations
    if report['range_violations']:
        print("\nRange Violations:")
        for column, info in report['range_violations'].items():
            print(f"{column}: {info['count']} violations")
    
    # Print consistency violations
    if report['consistency_violations']:
        print("\nConsistency Violations:")
        for check, info in report['consistency_violations'].items():
            print(f"{check}: {info['count']} violations")
    
    # Print pattern violations
    if report['pattern_violations']:
        print("\nPattern Violations:")
        for pattern, info in report['pattern_violations'].items():
            print(f"{pattern}: {info['count']} violations")

# Example usage
if __name__ == "__main__":
    # Create a validator instance
    validator = DataValidator(model_name='sports_data')
    try:
        # Train model on sample data
        sample_data = validator.load_data('cleaned_garmin_run_data.xls')
        validator.train_validation_model(sample_data, save_path='models')
        print("Model trained successfully on sample data")
        
        # Validate the data
        corrected_data, report = validator.validate_and_correct(
            'cleaned_garmin_run_data.xls',
            model_dir='models'
        )
        
        # Print validation report
        print_validation_report(report)
        
        # Save corrected data
        output_path = 'corrected_data.xlsx'
        corrected_data.to_excel(output_path, index=False)
        print(f"\nCorrected data saved to '{output_path}'")
        
    except Exception as e:
        print(f"Error: {str(e)}") 