import os
import json
import lmdb
import numpy as np
from glob import glob
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# Import functions from existing modules
from parse import read_records, extract_item9g_image, ETL9G_RECORD_SIZE
from clean import process_kanji_dict

def setup_lmdb(output_dir, map_size=1e9):
    """
    Set up the LMDB environment and return it along with an empty index.
    
    Args:
        output_dir: Directory to store the LMDB database
        map_size: Maximum size of the database in bytes (default: 1GB)
        
    Returns:
        tuple: (LMDB environment, character index defaultdict)
    """
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, 'kanji.lmdb')
    env = lmdb.open(db_path, map_size=int(map_size))
    index = defaultdict(int)
    return env, index

def process_and_store_kanji(kanji_dict, txn, index):
    """
    Process a kanji dictionary and store it in the LMDB database.
    
    Args:
        kanji_dict: Dictionary with character and image data
        txn: LMDB transaction object
        index: Character index defaultdict
        
    Returns:
        str: The key used to store the data
    """
    character = kanji_dict['character']
    
    # Get the black and white image
    bw_img = kanji_dict['twoBit']
    
    # Convert PIL image to bytes for storage
    img_bytes = bw_img.tobytes()
    
    # Create a unique key using the character, its index, and source
    key = f"{character}_{index[character]}_etl9g"
    
    # Store the image bytes directly in LMDB
    txn.put(key.encode(), img_bytes)
    
    # Update the index
    index[character] += 1
    
    return key

def process_etl9g_file(file_path, txn, index, limit=None):
    """
    Process a single ETL9G file and store the data in LMDB.
    
    Args:
        file_path: Path to an ETL9G data file
        txn: LMDB transaction object
        index: Character index defaultdict
        limit: Optional limit on the number of records to process
        
    Returns:
        int: Number of records processed
    """
    processed_count = 0
    
    print(f"Processing file: {file_path}")
    
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
                
                # Store in LMDB
                key = process_and_store_kanji(item_data, txn, index)
                processed_count += 1
                
                record_idx += 1
            except Exception as e:
                print(f"Error processing record {record_idx}: {e}")
                continue
    
    return processed_count

def prepare_etl9g_dataset(output_dir, limit=None):
    """
    Main function to prepare the ETL9G dataset and store in LMDB.
    
    Args:
        output_dir: Directory to store the LMDB database
        limit: Optional limit on the number of records to process per file
        
    Returns:
        tuple: (LMDB environment, total records processed)
    """
    # Find all ETL9G files
    image_files = sorted(glob('/data/ETL9G/ETL9G_*'))
    if not image_files:
        print("No ETL9G files found. Please check the path '/data/ETL9G/'.")
        return None, 0
    
    print(f"Found {len(image_files)} ETL9G files.")
    
    # Set up LMDB environment
    env, index = setup_lmdb(output_dir)
    total_processed = 0
    
    # Process each ETL9G file
    with env.begin(write=True) as txn:
        for file_idx, file_path in enumerate(tqdm(image_files, desc="Processing ETL9G files")):
            processed = process_etl9g_file(file_path, txn, index, limit)
            total_processed += processed
            print(f"Processed {processed} records from {file_path}")
    
    # Store the index in the database for future reference
    with env.begin(write=True) as txn:
        metadata = {
            'total_records': total_processed,
            'unique_characters': len(index),
            'character_counts': {k: v for k, v in index.items()}
        }
        txn.put(b'__metadata__', json.dumps(metadata).encode())
    
    print(f"Total records processed: {total_processed}")
    print(f"Total unique characters: {len(index)}")
    
    return env, total_processed

def main():
    """
    Main entry point for the script.
    """
    # Set up the output directory
    output_dir = '/app/output/prep'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting ETL9G dataset preparation with LMDB storage")
    
    # Process ETL9G files and store in LMDB
    env, total_processed = prepare_etl9g_dataset(output_dir)
    
    if env is None:
        print("Failed to prepare dataset. Exiting.")
        return
    
    # Close the LMDB environment
    env.close()
    
    print("\nDataset preparation complete!")
    print(f"Processed {total_processed} records")
    print(f"LMDB database saved to: {os.path.join(output_dir, 'kanji.lmdb')}")

if __name__ == "__main__":
    main()
