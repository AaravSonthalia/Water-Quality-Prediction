import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Define curve functions
def sinusoidal(x, A, B, C, D):
    """Sinusoidal function: y = A * sin(B * x + C) + D"""
    return A * np.sin(B * x + C) + D

def polynomial_3(x, a, b, c, d):
    """3rd degree polynomial: y = a*x³ + b*x² + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d

# Load fitted parameters
params_path = Path(__file__).parent / 'fitted_monthly_params.pkl'
with open(params_path, 'rb') as f:
    fitted_params = pickle.load(f)

print("="*80)
print("ADDING MONTH_FITTED FEATURE TO ALL DATASETS")
print("="*80)

# Display the equations being used
print("\nFitted Equations:")
print("-"*80)
print(f"TA:  {fitted_params['TA']['equation']}")
print(f"EC:  {fitted_params['EC']['equation']}")
print(f"DRP: {fitted_params['DRP']['equation']}")
print()

# Define datasets and their corresponding parameter types
datasets_config = {
    'ta_training.csv': {'param': 'TA', 'name': 'TA Training'},
    'ta_validation.csv': {'param': 'TA', 'name': 'TA Validation'},
    'ec_training.csv': {'param': 'EC', 'name': 'EC Training'},
    'ec_validation.csv': {'param': 'EC', 'name': 'EC Validation'},
    'drp_training.csv': {'param': 'DRP', 'name': 'DRP Training'},
    'drp_validation.csv': {'param': 'DRP', 'name': 'DRP Validation'}
}

data_path = Path(__file__).parent.parent.parent / 'New Datasets' / 'Combined' / 'Final Datasets'

# Process each dataset
for filename, config in datasets_config.items():
    print(f"\nProcessing: {config['name']} ({filename})")
    print("-"*80)
    
    # Load dataset
    file_path = data_path / filename
    df = pd.read_csv(file_path)
    
    print(f"  Original shape: {df.shape}")
    
    # Check if sample_date exists and extract month
    if 'sample_date' in df.columns:
        # Convert sample_date to datetime and extract month
        # Dates are already in ISO format (YYYY-MM-DD) from previous processing
        df['sample_date'] = pd.to_datetime(df['sample_date'])
        df['month'] = df['sample_date'].dt.month
    elif 'month_fitted' in df.columns:
        print(f"  ⚠ month_fitted already exists - skipping this dataset")
        continue
    else:
        print(f"  ⚠ WARNING: Neither sample_date nor month_fitted found - cannot process")
        continue
    
    # Get the appropriate model and parameters
    param_type = config['param']
    model_type = fitted_params[param_type]['model']
    params = fitted_params[param_type]['params']
    
    print(f"  Parameter type: {param_type}")
    print(f"  Using model: {model_type}")
    print(f"  Equation: {fitted_params[param_type]['equation']}")
    
    # Calculate month_fitted based on the model type
    if model_type == 'sinusoidal':
        df['month_fitted'] = sinusoidal(df['month'], *params)
    elif model_type == 'polynomial_3':
        df['month_fitted'] = polynomial_3(df['month'], *params)
    elif model_type == 'polynomial_2':
        # If polynomial_2 was used
        df['month_fitted'] = params[0] * df['month']**2 + params[1] * df['month'] + params[2]
    else:
        print(f"  WARNING: Unknown model type: {model_type}")
        continue
    
    # Drop the temporary 'month' column and 'sample_date' if they exist
    cols_to_drop = [col for col in ['month', 'sample_date'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    print(f"  New shape: {df.shape}")
    print(f"  month_fitted statistics:")
    print(f"    Mean: {df['month_fitted'].mean():.4f}")
    print(f"    Std: {df['month_fitted'].std():.4f}")
    print(f"    Min: {df['month_fitted'].min():.4f}")
    print(f"    Max: {df['month_fitted'].max():.4f}")
    
    # Save back to file
    df.to_csv(file_path, index=False)
    print(f"  ✓ Saved with month_fitted feature")

print("\n" + "="*80)
print("ALL DATASETS UPDATED SUCCESSFULLY")
print("="*80)
print("\nSummary:")
print("- TA datasets (training & validation): Used sinusoidal model")
print("- EC datasets (training & validation): Used sinusoidal model")
print("- DRP datasets (training & validation): Used polynomial_3 model")
print("\nAll datasets now contain the 'month_fitted' engineered feature!")
