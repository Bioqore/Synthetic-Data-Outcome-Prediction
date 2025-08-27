"""
Main Pipeline for Synthetic Data Outcome Prediction
==================================================

This module implements the complete pipeline for synthetic data prediction
using Gaussian-Copula model as shown in the schema diagram.

Pipeline Steps:
1. Load Historical Dataset (planned designs + actual outcomes)
2. Preprocess: Encode categorical columns (promoter, ori, etc.)
3. Fit Gaussian Copula: Use copulas.multivariate.GaussianMultivariate
4. Train Surrogate Model: RandomForestRegressor or Bayesian NN
5. Predict Outcomes: Use trained model on input parameter combinations
6. Postprocess & Decode: Convert encoded values back to readable names
7. Save Predictions: Combined inputs + predicted outcomes to CSV
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from enum import Enum

# Basic logging config and logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    BAYESIAN_NN = "bayesian_nn"
    GRADIENT_BOOSTING = "gradient_boosting"

class EncodingType(Enum):
    LABEL_ENCODING = "label"
    ONE_HOT_ENCODING = "onehot"
    TARGET_ENCODING = "target"

@dataclass
class PipelineConfig:
    # Data paths
    historical_data_path: str = "data/historical_data.csv"
    prediction_input_path: str = "data/prediction_input.csv"
    output_path: str = "output/synthetic_predictions.csv"

    # Model parameters
    surrogate_model_type: ModelType = ModelType.RANDOM_FOREST
    encoding_type: EncodingType = EncodingType.LABEL_ENCODING

    # Gaussian Copula parameters
    copula_random_state: int = 42
    n_synthetic_samples: int = 1000

    # Model hyperparameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_random_state: int = 42

    # Bayesian NN parameters (if using)
    nn_hidden_layers: List[int] = None
    nn_epochs: int = 100
    nn_batch_size: int = 32

    # Target columns
    target_columns: List[str] = None

    # Categorical columns
    categorical_columns: List[str] = None

    # ID column (to preserve during processing)
    id_column: Optional[str] = None

    # Validation parameters
    test_size: float = 0.2
    validation_random_state: int = 42

    def __post_init__(self):
        if self.target_columns is None:
            self.target_columns = ['yield', 'burden']
        if self.categorical_columns is None:
            self.categorical_columns = ['part_name', 'condition']
        if self.nn_hidden_layers is None:
            self.nn_hidden_layers = [64, 32, 16]


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    level = getattr(logging, log_level.upper())
    if log_file:
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def load_data(file_path: str) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    if p.suffix.lower() == '.csv':
        return pd.read_csv(p)
    if p.suffix.lower() in ['.xlsx', '.xls']:
        return pd.read_excel(p)
    if p.suffix.lower() == '.json':
        return pd.read_json(p)
    raise ValueError(f"Unsupported file format: {p.suffix}")


class DataValidator:
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True

    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        quality_report: Dict[str, Any] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': int(df.duplicated().sum()),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        }
        numeric_cols = quality_report['numeric_columns']
        outliers: Dict[str, int] = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = int(len(df[(df[col] < lower_bound) | (df[col] > upper_bound)]))
        quality_report['outliers'] = outliers
        logger.info(f"Data quality report: {quality_report}")
        return quality_report


class DataPreprocessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.is_fitted = False
        self.id_values: Optional[pd.Series] = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
        df_processed = df.copy()
        if self.config.id_column and self.config.id_column in df_processed.columns:
            self.id_values = df_processed[self.config.id_column].copy()
            df_processed = df_processed.drop(self.config.id_column, axis=1)
        if self.config.encoding_type == EncodingType.LABEL_ENCODING:
            for col in self.config.categorical_columns:
                if col in df_processed.columns:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.encoders[col] = le
        elif self.config.encoding_type == EncodingType.ONE_HOT_ENCODING:
            for col in self.config.categorical_columns:
                if col in df_processed.columns:
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = ohe.fit_transform(df_processed[[col]])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                        index=df_processed.index,
                    )
                    df_processed = pd.concat([df_processed.drop(col, axis=1), encoded_df], axis=1)
                    self.encoders[col] = ohe
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in self.config.target_columns]
        if feature_cols:
            scaler = StandardScaler()
            df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
            self.scalers['features'] = scaler
        self.is_fitted = True
        logger.info("Data preprocessing completed successfully")
        return df_processed

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        df_processed = df.copy()
        if self.config.id_column and self.config.id_column in df_processed.columns:
            self.id_values = df_processed[self.config.id_column].copy()
            df_processed = df_processed.drop(self.config.id_column, axis=1)
        if self.config.encoding_type == EncodingType.LABEL_ENCODING:
            for col in self.config.categorical_columns:
                if col in df_processed.columns and col in self.encoders:
                    unique_values = set(df_processed[col].astype(str))
                    known_values = set(self.encoders[col].classes_)
                    unseen_values = unique_values - known_values
                    if unseen_values:
                        logger.warning(f"Unseen categories in {col}: {unseen_values}")
                        df_processed[col] = df_processed[col].astype(str).replace(
                            list(unseen_values), [self.encoders[col].classes_[0]] * len(unseen_values)
                        )
                    df_processed[col] = self.encoders[col].transform(df_processed[col].astype(str))
        elif self.config.encoding_type == EncodingType.ONE_HOT_ENCODING:
            for col in self.config.categorical_columns:
                if col in df_processed.columns and col in self.encoders:
                    encoded = self.encoders[col].transform(df_processed[[col]])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[f"{col}_{cat}" for cat in self.encoders[col].categories_[0]],
                        index=df_processed.index,
                    )
                    df_processed = pd.concat([df_processed.drop(col, axis=1), encoded_df], axis=1)
        if 'features' in self.scalers:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in self.config.target_columns]
            feature_cols = [c for c in feature_cols if c in df_processed.columns]
            if feature_cols:
                df_processed[feature_cols] = self.scalers['features'].transform(df_processed[feature_cols])
        return df_processed

    def inverse_transform_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        df_decoded = df.copy()
        if self.config.encoding_type == EncodingType.LABEL_ENCODING:
            for col in self.config.categorical_columns:
                if col in df_decoded.columns and col in self.encoders:
                    df_decoded[col] = self.encoders[col].inverse_transform(df_decoded[col].astype(int))
        elif self.config.encoding_type == EncodingType.ONE_HOT_ENCODING:
            for col in self.config.categorical_columns:
                if col in self.encoders:
                    encoded_cols = [c for c in df_decoded.columns if c.startswith(f"{col}_")]
                    if encoded_cols:
                        encoded_values = df_decoded[encoded_cols].values
                        original_values = self.encoders[col].inverse_transform(encoded_values)
                        df_decoded[col] = original_values
                        df_decoded = df_decoded.drop(encoded_cols, axis=1)
        if self.id_values is not None and self.config.id_column:
            df_decoded[self.config.id_column] = self.id_values.reset_index(drop=True)
        return df_decoded


class ModelTrainer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model: Any = None
        self.is_fitted = False

    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.validation_random_state
        )
        if self.config.surrogate_model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=self.config.rf_random_state,
                n_jobs=-1,
            )
        elif self.config.surrogate_model_type == ModelType.GRADIENT_BOOSTING:
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=self.config.rf_random_state,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.surrogate_model_type}")
        logger.info(f"Training {self.config.surrogate_model_type.value} model...")
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        metrics: Dict[str, float] = {}
        for i, target_col in enumerate(self.config.target_columns):
            if y_val.shape[1] > 1:
                y_true_col = y_val.iloc[:, i]
                y_pred_col = y_pred[:, i]
            else:
                y_true_col = y_val.iloc[:, 0]
                y_pred_col = y_pred
            metrics[f"{target_col}_mse"] = float(mean_squared_error(y_true_col, y_pred_col))
            metrics[f"{target_col}_rmse"] = float(np.sqrt(metrics[f"{target_col}_mse"]))
            metrics[f"{target_col}_mae"] = float(mean_absolute_error(y_true_col, y_pred_col))
            metrics[f"{target_col}_r2"] = float(r2_score(y_true_col, y_pred_col))
        self.is_fitted = True
        logger.info(f"Model training completed. Metrics: {metrics}")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)


class OutputManager:
    @staticmethod
    def save_predictions(predictions: pd.DataFrame, output_path: str, include_metadata: bool = True) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        if include_metadata:
            import json
            metadata = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_predictions': len(predictions),
                'columns': list(predictions.columns),
                'summary_stats': predictions.describe().to_dict(),
            }
            metadata_path = output_path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Predictions saved to {output_path}")

    @staticmethod
    def generate_report(metrics: Dict[str, float], config: PipelineConfig, output_dir: str = "output") -> None:
        import json
        from datetime import datetime
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        report = {
            'execution_timestamp': datetime.now().isoformat(),
            'pipeline_config': {
                'surrogate_model_type': config.surrogate_model_type.value,
                'encoding_type': config.encoding_type.value,
                'n_synthetic_samples': config.n_synthetic_samples,
                'target_columns': config.target_columns,
                'categorical_columns': config.categorical_columns,
            },
            'model_performance': metrics,
            'data_paths': {
                'historical_data': config.historical_data_path,
                'prediction_input': config.prediction_input_path,
                'output': config.output_path,
            },
        }
        report_path = Path(output_dir) / 'pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Pipeline report saved to {report_path}")

class SyntheticDataPipeline:
    """
    Main pipeline class for synthetic data prediction using Gaussian-Copula model.
    
    This class orchestrates the entire workflow from data loading to prediction output.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.model_trainer = ModelTrainer(config)
        self.copula_model = None
        self.historical_data = None
        self.prediction_input = None
        
        # Setup logging
        setup_logging(log_file="pipeline.log")
        logger.info("Synthetic Data Pipeline initialized")
    
    def load_historical_data(self) -> pd.DataFrame:
        """
        Load historical dataset with planned designs and actual outcomes.
        
        Returns:
            Historical data DataFrame
        """
        logger.info(f"Loading historical data from {self.config.historical_data_path}")
        
        try:
            self.historical_data = load_data(self.config.historical_data_path)
            
            # Validate data
            required_columns = self.config.categorical_columns + self.config.target_columns
            if not DataValidator.validate_dataframe(self.historical_data, required_columns):
                raise ValueError("Historical data validation failed")
            
            # Perform data quality checks
            quality_report = DataValidator.check_data_quality(self.historical_data)
            
            logger.info(f"Historical data loaded successfully: {len(self.historical_data)} rows, {len(self.historical_data.columns)} columns")
            return self.historical_data
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            raise
    
    def load_prediction_input(self) -> pd.DataFrame:
        """
        Load prediction input set with parameter combinations.
        
        Returns:
            Prediction input DataFrame
        """
        logger.info(f"Loading prediction input from {self.config.prediction_input_path}")
        
        try:
            self.prediction_input = load_data(self.config.prediction_input_path)
            
            # Validate that categorical columns exist
            available_cat_cols = [col for col in self.config.categorical_columns 
                                if col in self.prediction_input.columns]
            if not available_cat_cols:
                logger.warning("No categorical columns found in prediction input")
            
            # Check if ID column exists
            if self.config.id_column and self.config.id_column not in self.prediction_input.columns:
                logger.warning(f"ID column '{self.config.id_column}' not found in prediction input")
            
            logger.info(f"Prediction input loaded successfully: {len(self.prediction_input)} rows, {len(self.prediction_input.columns)} columns")
            return self.prediction_input
            
        except Exception as e:
            logger.error(f"Error loading prediction input: {str(e)}")
            raise
    
    def fit_gaussian_copula(self, data: pd.DataFrame) -> None:
        """
        Fit Gaussian Copula model to capture joint dependencies.
        
        Args:
            data: Preprocessed historical data
        """
        logger.info("Fitting Gaussian Copula model...")
        
        try:
            from copulas.multivariate import GaussianMultivariate
            
            # Ensure all data is numeric for the copula model
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < data.shape[1]:
                logger.warning(f"Dropping non-numeric columns for Gaussian Copula fitting: {set(data.columns) - set(numeric_data.columns)}")
            
            # Initialize Gaussian Copula
            self.copula_model = GaussianMultivariate()
            
            # Fit the copula to the numeric data
            self.copula_model.fit(numeric_data)
            
            logger.info("Gaussian Copula model fitted successfully")
            
        except ImportError:
            logger.error("copulas library not found. Please install with: pip install copulas")
            raise
        except Exception as e:
            logger.error(f"Error fitting Gaussian Copula: {str(e)}")
            raise
    
    def generate_synthetic_samples(self, n_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic samples using the fitted Gaussian Copula.
        
        Args:
            n_samples: Number of synthetic samples to generate
            
        Returns:
            Synthetic samples DataFrame
        """
        if self.copula_model is None:
            raise ValueError("Copula model must be fitted before generating samples")
        
        n_samples = n_samples or self.config.n_synthetic_samples
        logger.info(f"Generating {n_samples} synthetic samples...")
        
        try:
            synthetic_data = self.copula_model.sample(n_samples)
            logger.info(f"Generated {len(synthetic_data)} synthetic samples")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generating synthetic samples: {str(e)}")
            raise
    
    def train_surrogate_model(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """
        Train surrogate model for outcome prediction.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame
            
        Returns:
            Training metrics dictionary
        """
        logger.info("Training surrogate model...")
        
        try:
            metrics = self.model_trainer.train(X, y)
            logger.info("Surrogate model training completed")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training surrogate model: {str(e)}")
            raise
    
    def predict_outcomes(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict outcomes using the trained surrogate model.
        
        Args:
            input_data: Input parameter combinations
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Making outcome predictions...")
        
        try:
            # Store original input data before preprocessing
            original_input = input_data.copy()
            
            # Preprocess input data
            processed_input = self.preprocessor.transform(input_data)
            
            # Remove target columns if they exist in input
            feature_cols = [col for col in processed_input.columns 
                          if col not in self.config.target_columns]
            X_pred = processed_input[feature_cols]
            
            # Make predictions
            predictions = self.model_trainer.predict(X_pred)
            
            # Create predictions DataFrame
            pred_df = pd.DataFrame(predictions, columns=self.config.target_columns)
            
            # Combine with original input data (this will include the ID column if it exists)
            if self.config.id_column and self.config.id_column in original_input.columns:
                # Ensure ID column is preserved
                result_df = pd.concat([original_input.reset_index(drop=True), pred_df], axis=1)
            else:
                # Standard case without ID column
                result_df = pd.concat([input_data.reset_index(drop=True), pred_df], axis=1)
            
            logger.info(f"Predictions completed for {len(result_df)} samples")
            return result_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def postprocess_and_decode(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Postprocess predictions and decode categorical values.
        
        Args:
            predictions: Raw predictions DataFrame
            
        Returns:
            Decoded predictions DataFrame
        """
        logger.info("Postprocessing and decoding predictions...")
        
        try:
            # Store ID column if it exists before decoding
            id_column_values = None
            if self.config.id_column and self.config.id_column in predictions.columns:
                id_column_values = predictions[self.config.id_column].copy()
            
            # For original input data, we don't need to decode categorical columns
            # since they were never encoded (we kept the original input)
            decoded_predictions = predictions.copy()
            
            # Ensure ID column is preserved if it wasn't handled by the preprocessor
            if id_column_values is not None and self.config.id_column not in decoded_predictions.columns:
                decoded_predictions[self.config.id_column] = id_column_values.reset_index(drop=True)
            
            # Round numeric predictions to reasonable precision
            for col in self.config.target_columns:
                if col in decoded_predictions.columns:
                    decoded_predictions[col] = decoded_predictions[col].round(4)
            
            logger.info("Postprocessing completed")
            return decoded_predictions
            
        except Exception as e:
            logger.error(f"Error in postprocessing: {str(e)}")
            raise
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Execute the complete synthetic data prediction pipeline.
        
        Returns:
            Final predictions DataFrame
        """
        logger.info("Starting synthetic data prediction pipeline...")
        
        try:
            # Step 1: Load Historical Dataset
            historical_data = self.load_historical_data()
            
            # Step 2: Preprocess Data (encode categorical columns)
            logger.info("Preprocessing historical data...")
            processed_historical = self.preprocessor.fit_transform(historical_data)
            
            # Step 3: Fit Gaussian Copula
            self.fit_gaussian_copula(processed_historical)
            
            # Step 4: Train Surrogate Model
            # Separate features and targets
            feature_cols = [col for col in processed_historical.columns 
                          if col not in self.config.target_columns]
            X_train = processed_historical[feature_cols]
            y_train = processed_historical[self.config.target_columns]
            
            training_metrics = self.train_surrogate_model(X_train, y_train)
            
            # Step 5: Load Prediction Input
            prediction_input = self.load_prediction_input()
            original_input = prediction_input.copy()
            
            # Step 6: Predict Outcomes
            predictions = self.predict_outcomes(prediction_input)
            
            # Step 7: Postprocess & Decode
            final_predictions = self.postprocess_and_decode(predictions)
            
            # Step 8: Create results file (input features + predictions)
            logger.info("Creating results file with input features and predictions...")
            
            # Extract only the prediction columns
            prediction_columns = self.config.target_columns
            
            # If plasmid_id column exists, use it to merge
            if self.config.id_column and self.config.id_column in original_input.columns and self.config.id_column in final_predictions.columns:
                logger.info(f"Merging based on {self.config.id_column} column")
                results_df = original_input.merge(
                    final_predictions[[self.config.id_column] + prediction_columns],
                    on=self.config.id_column,
                    how='left'
                )
            else:
                # If no id column, assume the rows are in the same order
                logger.warning("No ID column found. Assuming rows are in the same order.")
                results_df = original_input.copy()
                for col in prediction_columns:
                    if col in final_predictions.columns:
                        results_df[col] = final_predictions[col].values[:len(original_input)]
            
            # Ensure outcome columns are last
            ordered_cols = [c for c in results_df.columns if c not in self.config.target_columns] + self.config.target_columns
            results_df = results_df[ordered_cols]

            # Save results
            OutputManager.save_predictions(
                results_df, 
                self.config.output_path,
                include_metadata=True
            )
            
            # Generate comprehensive report
            OutputManager.generate_report(training_metrics, self.config)
            
            logger.info("Pipeline execution completed successfully!")
            return results_df
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def run_with_synthetic_augmentation(self) -> pd.DataFrame:
        """
        Execute pipeline with synthetic data augmentation.
        
        This method generates synthetic samples to augment the training data
        before training the surrogate model.
        
        Returns:
            Final predictions DataFrame
        """
        logger.info("Starting pipeline with synthetic data augmentation...")
        
        try:
            # Step 1-2: Same as regular pipeline
            historical_data = self.load_historical_data()
            processed_historical = self.preprocessor.fit_transform(historical_data)
            
            # Step 3: Fit Gaussian Copula (on numeric data only)
            numeric_processed = processed_historical.select_dtypes(include=[np.number])
            self.fit_gaussian_copula(numeric_processed)
            
            # Step 4: Generate synthetic samples
            synthetic_numeric = self.generate_synthetic_samples()
            
            # Combine synthetic numeric data with original categorical columns
            # This is needed because Gaussian Copula only works with numeric data
            logger.info("Preparing synthetic data with categorical features...")
            
            # Get categorical columns from processed historical data
            cat_cols = [col for col in processed_historical.columns if col not in numeric_processed.columns]
            
            # For synthetic data, randomly sample categorical values from historical data
            synthetic_samples = synthetic_numeric.copy()
            
            if cat_cols:
                n_synthetic = len(synthetic_samples)
                for col in cat_cols:
                    # Sample with replacement from historical categorical values
                    synthetic_samples[col] = np.random.choice(
                        processed_historical[col].values, 
                        size=n_synthetic, 
                        replace=True
                    )
            
            # Step 5: Combine historical and synthetic data
            logger.info("Combining historical and synthetic data...")
            combined_data = pd.concat([processed_historical, synthetic_samples], 
                                    axis=0, ignore_index=True)
            
            # Step 6: Train surrogate model on combined data
            feature_cols = [col for col in combined_data.columns 
                          if col not in self.config.target_columns]
            X_train = combined_data[feature_cols]
            y_train = combined_data[self.config.target_columns]
            
            training_metrics = self.train_surrogate_model(X_train, y_train)
            
            # Step 7: Load prediction input
            prediction_input = self.load_prediction_input()
            original_input = prediction_input.copy()
            
            # Step 8: Predict outcomes
            predictions = self.predict_outcomes(prediction_input)
            
            # Step 9: Postprocess & Decode
            final_predictions = self.postprocess_and_decode(predictions)
            
            # Step 10: Create results file (input features + predictions)
            logger.info("Creating results file with input features and predictions...")
            
            # Extract only the prediction columns
            prediction_columns = self.config.target_columns
            
            # If plasmid_id column exists, use it to merge
            if self.config.id_column and self.config.id_column in original_input.columns and self.config.id_column in final_predictions.columns:
                logger.info(f"Merging based on {self.config.id_column} column")
                results_df = original_input.merge(
                    final_predictions[[self.config.id_column] + prediction_columns],
                    on=self.config.id_column,
                    how='left'
                )
            else:
                # If no id column, assume the rows are in the same order
                logger.warning("No ID column found. Assuming rows are in the same order.")
                results_df = original_input.copy()
                for col in prediction_columns:
                    if col in final_predictions.columns:
                        results_df[col] = final_predictions[col].values[:len(original_input)]
            
            # Ensure outcome columns are last
            ordered_cols = [c for c in results_df.columns if c not in self.config.target_columns] + self.config.target_columns
            results_df = results_df[ordered_cols]

            # Save with augmented suffix if not already specified in the config
            if "_results" not in self.config.output_path:
                augmented_output_path = self.config.output_path.replace('.csv', '_augmented.csv')
            else:
                augmented_output_path = self.config.output_path
                
            OutputManager.save_predictions(
                results_df, 
                augmented_output_path,
                include_metadata=True
            )
            
            # Generate report
            OutputManager.generate_report(training_metrics, self.config, "output_augmented")
            
            logger.info("Pipeline with synthetic augmentation completed successfully!")
            return results_df
            
        except Exception as e:
            logger.error(f"Augmented pipeline execution failed: {str(e)}")
            raise
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """
        Validate the pipeline setup and data compatibility.
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating pipeline setup...")
        
        validation_results = {
            'data_paths_exist': True,
            'data_compatibility': True,
            'configuration_valid': True,
            'dependencies_available': True,
            'issues': []
        }
        
        try:
            # Check data paths
            if not Path(self.config.historical_data_path).exists():
                validation_results['data_paths_exist'] = False
                validation_results['issues'].append(f"Historical data not found: {self.config.historical_data_path}")
            
            if not Path(self.config.prediction_input_path).exists():
                validation_results['data_paths_exist'] = False
                validation_results['issues'].append(f"Prediction input not found: {self.config.prediction_input_path}")
            
            # Check dependencies
            try:
                import copulas
                import sklearn
            except ImportError as e:
                validation_results['dependencies_available'] = False
                validation_results['issues'].append(f"Missing dependency: {str(e)}")
            
            # Check configuration
            if not self.config.target_columns:
                validation_results['configuration_valid'] = False
                validation_results['issues'].append("No target columns specified")
            
            if not self.config.categorical_columns:
                validation_results['configuration_valid'] = False
                validation_results['issues'].append("No categorical columns specified")
            
            # If data exists, check compatibility
            if validation_results['data_paths_exist']:
                try:
                    hist_data = load_data(self.config.historical_data_path)
                    pred_input = load_data(self.config.prediction_input_path)
                    
                    # Check if required columns exist
                    missing_hist_cols = set(self.config.target_columns + self.config.categorical_columns) - set(hist_data.columns)
                    if missing_hist_cols:
                        validation_results['data_compatibility'] = False
                        validation_results['issues'].append(f"Missing columns in historical data: {missing_hist_cols}")
                    
                    missing_pred_cols = set(self.config.categorical_columns) - set(pred_input.columns)
                    if missing_pred_cols:
                        validation_results['data_compatibility'] = False
                        validation_results['issues'].append(f"Missing columns in prediction input: {missing_pred_cols}")
                    
                    # Check if ID column exists in prediction input if specified
                    if self.config.id_column and self.config.id_column not in pred_input.columns:
                        validation_results['data_compatibility'] = False
                        validation_results['issues'].append(f"ID column '{self.config.id_column}' not found in prediction input")
                        
                except Exception as e:
                    validation_results['data_compatibility'] = False
                    validation_results['issues'].append(f"Data loading error: {str(e)}")
            
            # Overall validation status
            validation_results['overall_valid'] = all([
                validation_results['data_paths_exist'],
                validation_results['data_compatibility'],
                validation_results['configuration_valid'],
                validation_results['dependencies_available']
            ])
            
            if validation_results['overall_valid']:
                logger.info("Pipeline validation passed")
            else:
                logger.warning(f"Pipeline validation failed: {validation_results['issues']}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during pipeline validation: {str(e)}")
            validation_results['overall_valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
            return validation_results

def create_sample_data(output_dir: str = "data") -> None:
    """
    Create sample datasets for testing the pipeline.
    
    Args:
        output_dir: Directory to save sample data files
    """
    logger.info("Creating sample datasets...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Sample historical data
    np.random.seed(42)
    n_samples = 200
    
    # Generate sample data
    part_names = ['PartA', 'PartB', 'PartC', 'PartD', 'PartE']
    conditions = ['Condition1', 'Condition2', 'Condition3']
    
    historical_data = pd.DataFrame({
        'part_name': np.random.choice(part_names, n_samples),
        'condition': np.random.choice(conditions, n_samples),
        'parameter1': np.random.normal(50, 10, n_samples),
        'parameter2': np.random.normal(100, 20, n_samples),
        'yield': np.random.normal(75, 15, n_samples),
        'burden': np.random.normal(25, 8, n_samples)
    })
    
    # Add some correlation between parameters and outcomes
    historical_data['yield'] += 0.3 * historical_data['parameter1'] + 0.2 * historical_data['parameter2']
    historical_data['burden'] -= 0.2 * historical_data['parameter1'] + 0.1 * historical_data['parameter2']
    
    # Ensure positive values
    historical_data['yield'] = np.clip(historical_data['yield'], 10, 150)
    historical_data['burden'] = np.clip(historical_data['burden'], 5, 50)
    
    # Sample prediction input
    prediction_input = pd.DataFrame({
        'part_name': np.random.choice(part_names, 50),
        'condition': np.random.choice(conditions, 50),
        'parameter1': np.random.normal(50, 10, 50),
        'parameter2': np.random.normal(100, 20, 50)
    })
    
    # Save sample data
    historical_data.to_csv(Path(output_dir) / "historical_data.csv", index=False)
    prediction_input.to_csv(Path(output_dir) / "prediction_input.csv", index=False)
    
    logger.info(f"Sample datasets created in {output_dir}/")
    logger.info(f"Historical data: {len(historical_data)} samples")
    logger.info(f"Prediction input: {len(prediction_input)} samples")

if __name__ == "__main__":
    # Example usage
    logger.info("Running Synthetic Data Prediction Pipeline")
    
    # Create sample data if it doesn't exist
    config = PipelineConfig()
    if not Path(config.historical_data_path).exists():
        logger.info("Sample data not found. Creating sample datasets...")
        create_sample_data()
    
    # Initialize and run pipeline
    pipeline = SyntheticDataPipeline(config)
    
    # Validate pipeline setup
    validation_results = pipeline.validate_pipeline()
    if not validation_results['overall_valid']:
        logger.error("Pipeline validation failed. Please check the issues and fix them.")
        for issue in validation_results['issues']:
            logger.error(f"- {issue}")
        exit(1)
    
    # Run the pipeline
    try:
        predictions = pipeline.run_pipeline()
        logger.info(f"Pipeline completed successfully! Predictions saved to {config.output_path}")
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Optionally run with synthetic augmentation
        # predictions_augmented = pipeline.run_with_synthetic_augmentation()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        exit(1) 