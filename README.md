# Synthetic Data Outcome Prediction Pipeline

A professional Python pipeline for synthetic data generation and outcome prediction using Gaussian-Copula models, designed for integration into production workflows.

## Overview

This pipeline implements a comprehensive synthetic data prediction system based on the following workflow:

1. **Load Historical Dataset** - Import planned designs with actual outcomes (CSV format)
2. **Preprocess Data** - Encode categorical columns (promoter, ori, etc.) using label/one-hot encoding
3. **Fit Gaussian Copula** - Use `copulas.multivariate.GaussianMultivariate` to model joint dependencies
4. **Train Surrogate Model** - RandomForestRegressor or Bayesian NN on historical data
5. **Predict Outcomes** - Use trained model to predict yield & burden on input parameter combinations
6. **Postprocess & Decode** - Convert encoded values back to readable part names
7. **Save Predictions** - Export combined inputs + predicted outcomes to CSV

## Features

- **Professional Architecture**: Modular design with clear separation of concerns
- **Flexible Configuration**: JSON-based configuration with command-line overrides
- **Multiple Model Types**: Support for Random Forest, Gradient Boosting, and Bayesian Neural Networks
- **Encoding Options**: Label encoding and one-hot encoding for categorical variables
- **Data Validation**: Comprehensive data quality checks and pipeline validation
- **Synthetic Augmentation**: Optional synthetic data generation to augment training sets
- **Comprehensive Logging**: Detailed logging with configurable levels
- **Error Handling**: Robust error handling with informative messages
- **Sample Data Generation**: Built-in sample data creation for testing

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone/Download the repository**
   ```bash
   # If using git
   git clone <repository-url>
   cd synthetic-data-outcome-predict
   
   # Or download and extract the files
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python run_pipeline.py --validate-only
   ```

## Quick Start

### 1. Create Sample Data (for testing)

```bash
python run_pipeline.py --create-sample-data
```

This creates sample datasets in the `data/` directory:
- `historical_data.csv` - Historical data with part names, conditions, parameters, and outcomes
- `prediction_input.csv` - Input parameter combinations for prediction

### 2. Run the Pipeline

```bash
# Run with default settings
python run_pipeline.py

# Run with synthetic data augmentation
python run_pipeline.py --augment

# Validate setup only (no execution)
python run_pipeline.py --validate-only
```

### 3. Check Results

The pipeline generates:
- `output/synthetic_predictions.csv` - Predictions with input parameters and predicted outcomes
- `output/synthetic_predictions_metadata.json` - Metadata about the predictions
- `output/pipeline_report.json` - Comprehensive execution report
- `pipeline.log` - Detailed execution logs

## Web Portal (Flask)

You can also run a simple web portal to upload CSVs and get predictions.

### Start the Web App

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

Then open your browser to `http://localhost:5000`.

### Portal Features
- Upload a training CSV (historical data with targets)
- Upload an input CSV (rows to predict)
- Optional checkbox for synthetic data augmentation
- Preview results and download the results CSV (input columns + predicted targets)

The app simply wraps the same pipeline you can run from the CLI.

## Configuration

### Default Configuration

The pipeline uses the following default settings:
- **Historical Data**: `data/historical_data.csv`
- **Prediction Input**: `data/prediction_input.csv`
- **Output**: `output/synthetic_predictions.csv`
- **Model Type**: Random Forest (100 estimators)
- **Encoding**: Label encoding
- **Target Columns**: `['yield', 'burden']`
- **Categorical Columns**: `['part_name', 'condition']`

### Custom Configuration

Create a JSON configuration file:

```json
{
  "historical_data_path": "data/my_historical_data.csv",
  "prediction_input_path": "data/my_prediction_input.csv",
  "output_path": "output/my_predictions.csv",
  "surrogate_model_type": "gradient_boosting",
  "encoding_type": "onehot",
  "n_synthetic_samples": 2000,
  "target_columns": ["yield", "burden", "efficiency"],
  "categorical_columns": ["part_name", "condition", "promoter"],
  "rf_n_estimators": 200,
  "rf_max_depth": 10,
  "test_size": 0.25
}
```

Run with custom configuration:
```bash
python run_pipeline.py --config my_config.json
```

## Command Line Interface

### Basic Usage

```bash
python run_pipeline.py [OPTIONS]
```

### Options

#### Data Paths
- `--historical-data PATH` - Path to historical data CSV file
- `--prediction-input PATH` - Path to prediction input CSV file
- `--output PATH` - Path for output predictions CSV file

#### Model Configuration
- `--model-type {random_forest,gradient_boosting}` - Type of surrogate model
- `--encoding-type {label,onehot}` - Type of categorical encoding
- `--n-samples N` - Number of synthetic samples to generate

#### Column Specifications
- `--target-columns COL1 COL2 ...` - List of target column names
- `--categorical-columns COL1 COL2 ...` - List of categorical column names

#### Execution Modes
- `--augment` - Run with synthetic data augmentation
- `--validate-only` - Only validate setup, don't run pipeline
- `--create-sample-data` - Create sample datasets for testing
- `--create-sample-config` - Create sample configuration file

#### Logging
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Logging level (default: INFO)
- `--log-file PATH` - Path to log file (default: console only)

### Examples

```bash
# Run with gradient boosting and one-hot encoding
python run_pipeline.py --model-type gradient_boosting --encoding-type onehot

# Run with custom data paths
python run_pipeline.py --historical-data my_data.csv --output my_results.csv

# Run with synthetic augmentation and detailed logging
python run_pipeline.py --augment --log-level DEBUG --log-file detailed.log

# Create sample configuration file
python run_pipeline.py --create-sample-config
```

## Data Format Requirements

### Historical Data (`historical_data.csv`)

Must contain:
- **Categorical columns**: As specified in configuration (default: `part_name`, `condition`)
- **Numeric feature columns**: Any additional parameters
- **Target columns**: As specified in configuration (default: `yield`, `burden`)

Example:
```csv
part_name,condition,parameter1,parameter2,yield,burden
PartA,Condition1,45.2,98.7,78.5,22.1
PartB,Condition2,52.8,105.3,82.1,18.9
...
```

### Prediction Input (`prediction_input.csv`)

Must contain:
- **Categorical columns**: Same as historical data
- **Numeric feature columns**: Same as historical data
- **No target columns**: These will be predicted

Example:
```csv
part_name,condition,parameter1,parameter2
PartA,Condition1,48.0,100.0
PartC,Condition3,55.2,110.5
...
```