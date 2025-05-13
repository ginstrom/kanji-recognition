import os
import pickle
import numpy as np
from glob import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Import functions from existing modules
from parse import read_records, extract_item9g_image, ETL9G_RECORD_SIZE
from clean import process_kanji_dict
from datasplit import split_dataset

def extract_etl9g_file_data(file_path, limit=None):
    """
    Extract data from a single ETL9G file and return a list of processed kanji dictionaries.
    
    Args:
        file_path: Path to an ETL9G data file
        limit: Optional limit on the number of records to process
        
    Returns:
        List of dictionaries, each with 'character', 'original', 'cropped', and 'twoBit' fields
    """
    kanji_dicts = []
    
    print(f"Processing file: {file_path}")
    
    # Read the binary file
    with open(file_path, 'rb') as f:
        record_idx = 0
        for s in read_records(f, ETL9G_RECORD_SIZE):
            if limit is not None and record_idx >= limit:
                break

            try:
                # Extract image and character data
                item_data = extract_item9g_image(s)
                
                if item_data is None:
                    print(f"Skipping record {record_idx} due to image data error.")
                    continue
                
                # Process the kanji dictionary to add the two-bit black and white image
                item_data = process_kanji_dict(item_data)
                
                # Add to our collection
                kanji_dicts.append(item_data)
                
                record_idx += 1
            except Exception as e:
                print(f"Error processing record {record_idx}: {e}")
                continue
    
    print(f"Extracted {len(kanji_dicts)} kanji images from {file_path}")
    return kanji_dicts

def prepare_etl9g_dataset(output_dir, limit=None):
    """
    Main function to prepare the ETL9G dataset for ML training.
    Processes one ETL9G file at a time and saves each to a separate pickle file.
    
    Args:
        output_dir: Directory to store the output files
        limit: Optional limit on the number of records to process per file
        
    Returns:
        List of paths to the pickle files containing processed data
    """
    # Find all ETL9G files
    image_files = sorted(glob('/data/ETL9G/ETL9G_*'))
    if not image_files:
        print("No ETL9G files found. Please check the path '/data/ETL9G/'.")
        return []
    
    print(f"Found {len(image_files)} ETL9G files.")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store paths to all generated pickle files
    pickle_files = []
    
    # Process each ETL9G file individually
    for file_idx, file_path in enumerate(image_files):
        # Extract file number from path (e.g., ETL9G_01 -> 01)
        file_name = os.path.basename(file_path)
        file_num = file_name.split('_')[-1]
        
        # Define pickle file path for this ETL9G file
        pickle_path = os.path.join(output_dir, f'etl9g_data_{file_num}.pkl')
        
        # Check if this file has already been processed
        if os.path.exists(pickle_path):
            print(f"Pickle file {pickle_path} already exists. Skipping processing of {file_path}.")
            pickle_files.append(pickle_path)
            continue
        
        # Extract and process data from this file
        kanji_dicts = extract_etl9g_file_data(file_path, limit)
        
        if not kanji_dicts:
            print(f"No kanji data extracted from {file_path}. Skipping.")
            continue
        
        # Save data to a pickle file
        print(f"Saving data from {file_path} to {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(kanji_dicts, f)
        
        print(f"Saved {len(kanji_dicts)} kanji images to {pickle_path}")
        pickle_files.append(pickle_path)
    
    if not pickle_files:
        print("No data was processed. Aborting.")
        return []
    
    print(f"Successfully processed {len(pickle_files)} ETL9G files.")
    return pickle_files

def split_and_save_datasets(pickle_files, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Split the dataset into train, validation, and test sets and save them.
    Loads data from multiple pickle files.
    
    Args:
        pickle_files: List of paths to pickle files containing processed data
        output_dir: Directory to store the output files
        train_ratio: Fraction of data to use for training
        val_ratio: Fraction of data to use for validation
        
    Returns:
        Dictionary with paths to the train, validation, and test pickle files
    """
    # Load data from all pickle files
    all_kanji_dicts = []
    
    print(f"Loading data from {len(pickle_files)} pickle files")
    for pickle_path in pickle_files:
        print(f"Loading data from {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                kanji_dicts = pickle.load(f)
                all_kanji_dicts.extend(kanji_dicts)
        except Exception as e:
            print(f"Error loading data from {pickle_path}: {e}")
            continue
    
    if not all_kanji_dicts:
        print("No data loaded. Cannot split dataset.")
        return {}
    
    print(f"Loaded {len(all_kanji_dicts)} kanji images in total")
    
    # Split the dataset
    print("Splitting dataset into train, validation, and test sets")
    split_indices = split_dataset(all_kanji_dicts, train_ratio, val_ratio)
    
    # Create the split datasets
    train_data = [all_kanji_dicts[i] for i in split_indices['train']]
    val_data = [all_kanji_dicts[i] for i in split_indices['val']]
    test_data = [all_kanji_dicts[i] for i in split_indices['test']]
    
    # Save the split datasets
    train_path = os.path.join(output_dir, 'etl9g_train.pkl')
    val_path = os.path.join(output_dir, 'etl9g_val.pkl')
    test_path = os.path.join(output_dir, 'etl9g_test.pkl')
    
    print(f"Saving train set ({len(train_data)} samples) to {train_path}")
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    
    print(f"Saving validation set ({len(val_data)} samples) to {val_path}")
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    
    print(f"Saving test set ({len(test_data)} samples) to {test_path}")
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)
    
    return {
        'train': train_path,
        'val': val_path,
        'test': test_path
    }

def main():
    """
    Main entry point for the script.
    """
    # Set up the output directory
    output_dir = '/app/output/prep'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting ETL9G dataset preparation")
    
    # Process ETL9G files individually and save to separate pickle files
    pickle_files = prepare_etl9g_dataset(output_dir)
    
    if not pickle_files:
        print("Failed to prepare dataset. Exiting.")
        return
    
    # Split the dataset and save the splits
    split_paths = split_and_save_datasets(pickle_files, output_dir)
    
    if not split_paths:
        print("Failed to split dataset. Exiting.")
        return
    
    print("\nDataset preparation complete!")
    print(f"Processed data saved to {len(pickle_files)} pickle files:")
    for path in pickle_files:
        print(f"  - {path}")
    print(f"Train set saved to: {split_paths.get('train', 'N/A')}")
    print(f"Validation set saved to: {split_paths.get('val', 'N/A')}")
    print(f"Test set saved to: {split_paths.get('test', 'N/A')}")

if __name__ == "__main__":
    main()
