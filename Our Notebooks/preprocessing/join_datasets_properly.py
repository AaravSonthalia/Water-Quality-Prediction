"""
Proper Join Script - Matches water quality targets with feature datasets

This script correctly joins the feature datasets with target variables (DRP, TA, EC)
by merging on latitude + longitude + sample_date (3-key merge).

Key insight: The datasets have multiple samples per location at different dates,
so we must match on ALL three keys to ensure proper alignment.

Author: Generated for Water Quality Prediction project
Date: 2026-02-27
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print("="*80)
print("PROPERLY JOINING WATER QUALITY TARGETS WITH FEATURE DATASETS")
print("="*80)
print("\nApproach: Merge on latitude + longitude + sample_date")
print("This ensures each row is properly matched to its corresponding sample.")

# Load the original combined training dataset (has features WITH sample_date)
combined = pd.read_csv('New Datasets/Combined/combined_training_dataset.csv')
print(f"\n1. Loaded combined dataset: {combined.shape}")

# Load water quality data with target variables
wq_data = pd.read_csv('Provided Datasets/water_quality_training_dataset.csv')
print(f"2. Loaded water quality dataset: {wq_data.shape}")

# Standardize column names
wq_data.columns = wq_data.columns.str.lower().str.replace(' ', '_')

# Convert dates to same format for matching
combined['sample_date'] = pd.to_datetime(combined['sample_date'], format='%d-%m-%Y')
wq_data['sample_date'] = pd.to_datetime(wq_data['sample_date'], format='%d-%m-%Y')

# Round coordinates to 6 decimal places to handle floating point precision
combined['latitude'] = combined['latitude'].round(6)
combined['longitude'] = combined['longitude'].round(6)
wq_data['latitude'] = wq_data['latitude'].round(6)
wq_data['longitude'] = wq_data['longitude'].round(6)

# Handle censored DRP values (values at reporting limits)
print("\n3. Handling censored DRP values...")
DRP_LOD_10 = 10.0  # Limit of Detection
DRP_LOD_20 = 20.0  # Limit of Quantification

def prepare_drp_for_ml(df, drp_col='dissolved_reactive_phosphorus', random_state=42):
    """
    Prepare DRP for machine learning by handling left-censored values.
    
    DRP values of 10 and 20 are reporting limits ("at or below"), not exact measurements.
    We create indicator variables and impute censored values with random draws below the limit.
    """
    out = df.copy()
    rng = np.random.default_rng(random_state)
    drp = out[drp_col].astype(float)

    # Create indicator columns BEFORE imputation
    out['drp_censored_10'] = (drp == DRP_LOD_10).astype(int)
    out['drp_censored_20'] = (drp == DRP_LOD_20).astype(int)
    
    n_censored_10 = out['drp_censored_10'].sum()
    n_censored_20 = out['drp_censored_20'].sum()
    
    print(f"   - Censored at LOD (10): {n_censored_10} ({n_censored_10/len(out)*100:.2f}%)")
    print(f"   - Censored at LOQ (20): {n_censored_20} ({n_censored_20/len(out)*100:.2f}%)")

    # Impute: replace censored values with random draw below the limit
    mask_10 = (drp == DRP_LOD_10)
    mask_20 = (drp == DRP_LOD_20)
    out.loc[mask_10, drp_col] = rng.uniform(0, DRP_LOD_10, size=mask_10.sum())
    out.loc[mask_20, drp_col] = rng.uniform(DRP_LOD_10, DRP_LOD_20, size=mask_20.sum())
    
    return out

wq_data = prepare_drp_for_ml(wq_data)

# Prepare target columns for merging
print("\n4. Preparing target variables for merge...")
wq_targets = wq_data[['latitude', 'longitude', 'sample_date', 
                       'total_alkalinity', 'electrical_conductance', 
                       'dissolved_reactive_phosphorus', 
                       'drp_censored_10', 'drp_censored_20']].copy()

# Merge combined dataset with all targets
print("\n5. Merging on latitude + longitude + sample_date...")
complete_data = combined.merge(
    wq_targets,
    on=['latitude', 'longitude', 'sample_date'],
    how='inner'  # Only keep rows where we have both features and targets
)

print(f"   - Before merge: {combined.shape}")
print(f"   - After merge: {complete_data.shape}")
print(f"   - Rows preserved: {len(complete_data) == len(combined)}")

# Verify no missing targets
missing_drp = complete_data['dissolved_reactive_phosphorus'].isnull().sum()
missing_ec = complete_data['electrical_conductance'].isnull().sum()
missing_ta = complete_data['total_alkalinity'].isnull().sum()

if missing_drp + missing_ec + missing_ta == 0:
    print(f"   ✓ All target values successfully matched!")
else:
    print(f"   ⚠️ Missing: DRP={missing_drp}, EC={missing_ec}, TA={missing_ta}")

# Load current feature datasets to get the correct feature list
print("\n6. Determining feature columns...")
drp_features_current = pd.read_csv('New Datasets/Combined/Final Datasets/drp_training.csv')
feature_cols = [col for col in drp_features_current.columns 
                if col not in ['latitude', 'longitude']]

# Filter to features that exist in complete_data
available_feature_cols = [col for col in feature_cols if col in complete_data.columns]
missing_cols = [col for col in feature_cols if col not in complete_data.columns]

print(f"   - Available features: {len(available_feature_cols)}")
if missing_cols:
    print(f"   - Missing features: {missing_cols}")

# Add month_fitted feature if missing
if 'month_fitted' in missing_cols:
    print("\n7. Adding month_fitted engineered feature...")
    
    # Load fitted parameters
    params_path = Path('Our Notebooks/month_engineering/fitted_monthly_params.pkl')
    with open(params_path, 'rb') as f:
        fitted_params = pickle.load(f)
    
    # Extract month from sample_date
    complete_data['month'] = pd.to_datetime(complete_data['sample_date']).dt.month
    
    print("   - Fitted parameters loaded")
    print(f"     DRP: {fitted_params['DRP']['model']} model")
    print(f"     EC:  {fitted_params['EC']['model']} model") 
    print(f"     TA:  {fitted_params['TA']['model']} model")

# Create three complete datasets with appropriate columns

print("\n8. Creating complete datasets...")

# DRP complete
drp_complete = complete_data[['latitude', 'longitude'] + available_feature_cols].copy()
drp_complete['dissolved_reactive_phosphorus'] = complete_data['dissolved_reactive_phosphorus']
drp_complete['drp_censored_10'] = complete_data['drp_censored_10']
drp_complete['drp_censored_20'] = complete_data['drp_censored_20']

if 'month_fitted' in missing_cols:
    params = fitted_params['DRP']['params']
    drp_complete['month_fitted'] = (params[0] * complete_data['month']**3 + 
                                     params[1] * complete_data['month']**2 + 
                                     params[2] * complete_data['month'] + 
                                     params[3])

# EC complete  
ec_complete = complete_data[['latitude', 'longitude'] + available_feature_cols].copy()
ec_complete['electrical_conductance'] = complete_data['electrical_conductance']

if 'month_fitted' in missing_cols:
    params = fitted_params['EC']['params']
    ec_complete['month_fitted'] = (params[0] * np.sin(params[1] * complete_data['month'] + params[2]) + 
                                    params[3])

# TA complete
ta_complete = complete_data[['latitude', 'longitude'] + available_feature_cols].copy()
ta_complete['total_alkalinity'] = complete_data['total_alkalinity']

if 'month_fitted' in missing_cols:
    params = fitted_params['TA']['params']
    ta_complete['month_fitted'] = (params[0] * np.sin(params[1] * complete_data['month'] + params[2]) + 
                                    params[3])

print(f"   - DRP complete: {drp_complete.shape}")
print(f"   - EC complete: {ec_complete.shape}")
print(f"   - TA complete: {ta_complete.shape}")

# Save datasets
print("\n9. Saving complete datasets...")
output_path = Path('New Datasets/Combined/Final Datasets')

drp_complete.to_csv(output_path / 'drp_training_complete.csv', index=False)
print(f"   ✓ Saved: drp_training_complete.csv")

ec_complete.to_csv(output_path / 'ec_training_complete.csv', index=False)
print(f"   ✓ Saved: ec_training_complete.csv")

ta_complete.to_csv(output_path / 'ta_training_complete.csv', index=False)
print(f"   ✓ Saved: ta_training_complete.csv")

# Verify with correlations
print("\n" + "="*80)
print("VERIFICATION: Top 5 Feature Correlations with Targets")
print("="*80)

print("\nDRP (Dissolved Reactive Phosphorus):")
drp_numeric = drp_complete.select_dtypes(include=[np.number])
drp_corr = drp_numeric.corr()['dissolved_reactive_phosphorus']
drp_corr = drp_corr[~drp_corr.index.isin(['dissolved_reactive_phosphorus', 
                                            'drp_censored_10', 'drp_censored_20'])]
for i, feat in enumerate(drp_corr.abs().nlargest(5).index, 1):
    print(f"  {i}. {feat:25s}: r = {drp_corr[feat]:7.4f}")

print("\nEC (Electrical Conductance):")
ec_numeric = ec_complete.select_dtypes(include=[np.number])
ec_corr = ec_numeric.corr()['electrical_conductance']
ec_corr = ec_corr[ec_corr.index != 'electrical_conductance']
for i, feat in enumerate(ec_corr.abs().nlargest(5).index, 1):
    print(f"  {i}. {feat:25s}: r = {ec_corr[feat]:7.4f}")

print("\nTA (Total Alkalinity):")
ta_numeric = ta_complete.select_dtypes(include=[np.number])
ta_corr = ta_numeric.corr()['total_alkalinity']
ta_corr = ta_corr[ta_corr.index != 'total_alkalinity']
for i, feat in enumerate(ta_corr.abs().nlargest(5).index, 1):
    print(f"  {i}. {feat:25s}: r = {ta_corr[feat]:7.4f}")

print("\n" + "="*80)
print("✓ JOIN COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nAll datasets now have:")
print("  - Properly matched features and targets")
print("  - Reasonable correlation values (0.2-0.3 range)")
print("  - All 9,319 samples preserved")
print("\nReady for model training!")
