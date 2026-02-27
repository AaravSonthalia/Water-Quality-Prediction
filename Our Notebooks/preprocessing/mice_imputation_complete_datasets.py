"""
MICE Imputation for Complete Training Datasets

Applies Multivariate Imputation by Chained Equations (MICE) using IterativeImputer
with RandomForest to handle missing values in the Landsat features.

Missing data: ~11.64% of rows have missing Landsat data (likely due to cloud cover)
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MICE IMPUTATION FOR COMPLETE TRAINING DATASETS")
print("="*80)

data_path = Path('New Datasets/Combined/Final Datasets')

# Load datasets
drp_complete = pd.read_csv(data_path / 'drp_training_complete.csv')
ec_complete = pd.read_csv(data_path / 'ec_training_complete.csv')
ta_complete = pd.read_csv(data_path / 'ta_training_complete.csv')

print(f"\nLoaded datasets:")
print(f"  DRP: {drp_complete.shape}")
print(f"  EC: {ec_complete.shape}")
print(f"  TA: {ta_complete.shape}")

def impute_dataset(df, dataset_name, random_state=42):
    """
    Impute missing values using IterativeImputer with RandomForest
    """
    print(f"\n{'='*80}")
    print(f"Imputing: {dataset_name}")
    print(f"{'='*80}")
    
    # Separate metadata and target columns that shouldn't be imputed
    metadata_cols = ['latitude', 'longitude']
    
    # Get target columns (different for each dataset)
    target_cols = []
    if 'dissolved_reactive_phosphorus' in df.columns:
        target_cols = ['dissolved_reactive_phosphorus', 'drp_censored_10', 'drp_censored_20']
    elif 'electrical_conductance' in df.columns:
        target_cols = ['electrical_conductance']
    elif 'total_alkalinity' in df.columns:
        target_cols = ['total_alkalinity']
    
    # Get feature columns (exclude metadata and targets)
    feature_cols = [col for col in df.columns if col not in metadata_cols + target_cols]
    
    # Check for missing values
    total_missing = df[feature_cols].isnull().sum().sum()
    
    if total_missing == 0:
        print("✓ No missing values found - no imputation needed!")
        return df.copy()
    
    print(f"\nBefore imputation:")
    print(f"  Total missing values in features: {total_missing:,}")
    print(f"  Missing percentage: {(total_missing / (len(df) * len(feature_cols))) * 100:.2f}%")
    
    # Features with missing values
    features_with_missing = df[feature_cols].columns[df[feature_cols].isnull().any()].tolist()
    print(f"  Features with missing values: {len(features_with_missing)}")
    for feat in features_with_missing:
        missing_count = df[feat].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        print(f"    - {feat}: {missing_count} ({missing_pct:.2f}%)")
    
    # Create a copy for imputation
    df_imputed = df.copy()
    
    # Initialize IterativeImputer with RandomForest estimator
    # RandomForest is robust and can capture non-linear relationships
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=10, 
            random_state=random_state, 
            n_jobs=-1,
            max_depth=10  # Limit depth to prevent overfitting
        ),
        max_iter=10,
        random_state=random_state,
        verbose=0
    )
    
    print("\n⏳ Performing iterative imputation with RandomForest...")
    print("   (This may take a few minutes...)")
    
    # Impute only the feature columns
    df_imputed[feature_cols] = imputer.fit_transform(df[feature_cols])
    
    print("✓ Imputation complete!")
    
    # Verify no missing values remain in features
    remaining_missing = df_imputed[feature_cols].isnull().sum().sum()
    print(f"\nAfter imputation:")
    print(f"  Remaining missing values in features: {remaining_missing}")
    
    if remaining_missing == 0:
        print("  ✓ All missing values successfully imputed!")
    else:
        print(f"  ⚠ Warning: {remaining_missing} missing values still remain")
    
    # Show statistics for imputed features
    print(f"\n  Statistics for imputed features:")
    for feat in features_with_missing:
        original_mean = df[feat].mean()
        imputed_mean = df_imputed[feat].mean()
        print(f"    {feat}: original mean={original_mean:.2f}, after imputation={imputed_mean:.2f}")
    
    return df_imputed

# Impute all three datasets
print("\n" + "="*80)
print("STARTING IMPUTATION PROCESS")
print("="*80)

drp_imputed = impute_dataset(drp_complete, 'DRP Training Complete')
ec_imputed = impute_dataset(ec_complete, 'EC Training Complete')
ta_imputed = impute_dataset(ta_complete, 'TA Training Complete')

# Save imputed datasets
print("\n" + "="*80)
print("SAVING IMPUTED DATASETS")
print("="*80)

drp_imputed.to_csv(data_path / 'drp_training_complete.csv', index=False)
print(f"✓ Saved: drp_training_complete.csv ({drp_imputed.shape})")

ec_imputed.to_csv(data_path / 'ec_training_complete.csv', index=False)
print(f"✓ Saved: ec_training_complete.csv ({ec_imputed.shape})")

ta_imputed.to_csv(data_path / 'ta_training_complete.csv', index=False)
print(f"✓ Saved: ta_training_complete.csv ({ta_imputed.shape})")

# Verify no missing values
print("\n" + "="*80)
print("FINAL VERIFICATION")
print("="*80)

drp_missing = drp_imputed.isnull().sum().sum()
ec_missing = ec_imputed.isnull().sum().sum()
ta_missing = ta_imputed.isnull().sum().sum()

print(f"\nMissing values after imputation:")
print(f"  DRP: {drp_missing}")
print(f"  EC: {ec_missing}")
print(f"  TA: {ta_missing}")

if drp_missing + ec_missing + ta_missing == 0:
    print("\n✓ SUCCESS: All datasets fully imputed with no missing values!")
else:
    print(f"\n⚠️  WARNING: {drp_missing + ec_missing + ta_missing} missing values remain")

print("\n" + "="*80)
print("MICE IMPUTATION COMPLETED")
print("="*80)
print("\nAll three complete training datasets now have:")
print("  - No missing values")
print("  - Imputed Landsat features using RandomForest-based MICE")
print("  - Preserved all target variables and metadata")
print("\nReady for model training!")
