import pandas as pd
import os

# Define the columns to remove
columns_to_remove = [
    'gaia_impervious_frac_by_sample_year',
    'gsw_occurrence',
    'gsw_transitions'
]

# Define the folder and files
folder_path = 'New Datasets/Combined/Final Datasets'
csv_files = [
    'drp_training.csv',
    'drp_validation.csv',
    'ec_training.csv',
    'ec_validation.csv',
    'ta_training.csv',
    'ta_validation.csv'
]

# Process each file
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    print(f"Processing {csv_file}...")
    
    # Read the CSV
    df = pd.read_csv(file_path)
    
    # Get columns that exist in the dataframe and should be removed
    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    print(f"  Original shape: {df.shape}")
    print(f"  Removing columns: {existing_columns_to_remove}")
    
    # Drop the columns
    df = df.drop(columns=existing_columns_to_remove)
    
    print(f"  New shape: {df.shape}")
    
    # Save back to the same file
    df.to_csv(file_path, index=False)
    print(f"  Saved cleaned {csv_file}\n")

print("All files cleaned successfully!")
