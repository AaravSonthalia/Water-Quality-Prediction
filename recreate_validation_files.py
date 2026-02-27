import pandas as pd
import os

print("="*60)
print("RECREATING VALIDATION FILES WITH SAMPLE_DATE")
print("="*60)

# Load the original combined validation dataset (has sample_date)
combined_val = pd.read_csv('New Datasets/Combined/combined_validation_dataset.csv')
print(f"\nOriginal combined validation shape: {combined_val.shape}")
print(f"Columns: {list(combined_val.columns)}")

# Verify sample_date exists
if 'sample_date' not in combined_val.columns:
    print("ERROR: sample_date not found in combined validation!")
    exit(1)

print(f"\n✓ sample_date column found")
print(f"Unique locations: {combined_val[['latitude', 'longitude']].drop_duplicates().shape[0]}")
print(f"Total samples: {len(combined_val)}")

# Define the base features to keep (same as training)
base_features = ['latitude', 'longitude', 'sample_date', 'month_fitted', 
                 'swir16', 'swir22', 'red', 'NDMI', 'MNDWI',
                 'pet', 'aet', 'def', 'q', 'ppt', 'soil', 
                 'srad', 'tmax', 'tmin', 'vap', 'vpd', 'ws', 'pdsi',
                 'esa_lccs_class', 'esa_change_count']

# Check which features exist
available_features = [f for f in base_features if f in combined_val.columns]
missing_features = [f for f in base_features if f not in combined_val.columns]

print(f"\nAvailable features: {len(available_features)}")
if missing_features:
    print(f"Missing features: {missing_features}")

# Create validation datasets for each parameter
output_dir = 'New Datasets/Combined/Final Datasets'
os.makedirs(output_dir, exist_ok=True)

for param_name, param_file in [
    ('EC', 'ec_validation.csv'),
    ('DRP', 'drp_validation.csv'),
    ('TA', 'ta_validation.csv')
]:
    # Select features
    val_data = combined_val[available_features].copy()
    
    # Drop rows with missing values
    before_drop = len(val_data)
    val_data = val_data.dropna()
    after_drop = len(val_data)
    
    if before_drop != after_drop:
        print(f"\n{param_name}: Dropped {before_drop - after_drop} rows with missing values")
    
    # Save
    output_path = os.path.join(output_dir, param_file)
    val_data.to_csv(output_path, index=False)
    
    print(f"\n{param_name} Validation:")
    print(f"  File: {param_file}")
    print(f"  Shape: {val_data.shape}")
    print(f"  Columns: {list(val_data.columns)[:5]}... (+{len(val_data.columns)-5} more)")
    print(f"  ✓ Saved to: {output_path}")

print("\n" + "="*60)
print("VALIDATION FILES RECREATED SUCCESSFULLY")
print("="*60)
print("\nAll validation files now include:")
print("  - latitude, longitude")
print("  - sample_date (CRITICAL for matching)")
print("  - month_fitted")
print("  - All other features")
print("\nNow re-run your model notebooks to generate predictions.")
