"""
Drop SWE (Snow Water Equivalent) Column from All Datasets

Removes the 'swe' column from all training and validation datasets.
"""

import pandas as pd
from pathlib import Path

print("="*80)
print("DROPPING SWE COLUMN FROM ALL DATASETS")
print("="*80)

data_path = Path('New Datasets/Combined/Final Datasets')

# Define all datasets to process
datasets = {
    'drp_training_complete.csv': 'DRP Training Complete',
    'ec_training_complete.csv': 'EC Training Complete',
    'ta_training_complete.csv': 'TA Training Complete',
    'drp_training.csv': 'DRP Training',
    'ec_training.csv': 'EC Training',
    'ta_training.csv': 'TA Training',
    'drp_validation.csv': 'DRP Validation',
    'ec_validation.csv': 'EC Validation',
    'ta_validation.csv': 'TA Validation'
}

processed = 0
skipped = 0

for filename, dataset_name in datasets.items():
    file_path = data_path / filename
    
    if not file_path.exists():
        print(f"\n⚠️  Skipping {dataset_name}: File not found")
        skipped += 1
        continue
    
    # Load dataset
    df = pd.read_csv(file_path)
    original_shape = df.shape
    
    # Check if swe column exists
    if 'swe' in df.columns:
        # Drop swe column
        df = df.drop(columns=['swe'])
        
        # Save back
        df.to_csv(file_path, index=False)
        
        print(f"\n✓ {dataset_name}")
        print(f"  Original: {original_shape}")
        print(f"  After dropping 'swe': {df.shape}")
        print(f"  File: {filename}")
        processed += 1
    else:
        print(f"\n⚠️  {dataset_name}: 'swe' column not found (already removed?)")
        print(f"  Shape: {df.shape}")
        skipped += 1

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nProcessed: {processed} datasets")
print(f"Skipped: {skipped} datasets")

if processed > 0:
    print("\n✓ SWE column successfully removed from all applicable datasets!")
else:
    print("\n⚠️  No datasets were modified")
