"""
# ETL9G Dataset Preparation Script
This script processes the ETL9G dataset, extracts kanji characters and their corresponding images, and stores them in an LMDB database. It also handles metadata storage and error management during processing.
"""
import os
import json
import lmdb
import numpy as np
from glob import glob
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# Import functions from existing modules
from parse_etl9g import extract_images

def setup_lmdb(output_dir, map_size=20e9):
    """
    Set up the LMDB environment and return it along with an empty index.
    
    Args:
        output_dir: Directory to store the LMDB database
        map_size: Maximum size of the database in bytes (default: 20GB)
        
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

def process_etl9g_files(file_paths, env, index, limit=None):
    """
    Process ETL9G files and store the data in LMDB.
    
    Args:
        file_paths: List of paths to ETL9G data files
        env: LMDB environment 
        index: Character index defaultdict
        limit: Optional limit on the number of records to process per file
        
    Returns:
        int: Number of records processed
    """
    processed_count = 0
    
    # Use the generator function from parse_etl9g.py to get items one by one
    for item_data in extract_images(file_paths, limit):
        # Process each item in its own transaction to avoid MDB_BAD_TXN errors
        with env.begin(write=True) as txn:
            try:
                key = process_and_store_kanji(item_data, txn, index)
                processed_count += 1
                
                # Print progress every 100 items
                if processed_count % 1_000 == 0:
                    print(f"Processed {processed_count} items so far")
            except Exception as e:
                print(f"Error processing item: {e}")
    
    return processed_count

def save_metadata(env, total_processed, index):
    """
    Save metadata about the dataset to the LMDB database.
    
    Args:
        env: LMDB environment
        total_processed: Total number of records processed
        index: Character index defaultdict
    """
    try:
        with env.begin(write=True) as txn:
            metadata = {
                'total_records': total_processed,
                'unique_characters': len(index),
                'character_counts': {k: v for k, v in index.items()}
            }
            txn.put(b'__metadata__', json.dumps(metadata).encode())
            print("Metadata successfully saved to LMDB")
    except Exception as e:
        print(f"Error saving metadata: {e}")
        # Write to a separate file as backup
        metadata_path = os.path.join(os.path.dirname(env.path()), 'metadata.json')
        with open(metadata_path, 'w') as f:
            metadata = {
                'total_records': total_processed,
                'unique_characters': len(index),
                'sources': ['etl9g'],
                'character_counts_etl9g': {k: v for k, v in index.items()}
            }
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"Metadata saved to {metadata_path} as backup")

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
    
    # Set up LMDB environment with larger map_size to avoid MDB_MAP_FULL errors
    env, index = setup_lmdb(output_dir, map_size=20e9)  # 20GB map size
    
    # Initialize total_processed before the try block to avoid UnboundLocalError
    total_processed = 0
    
    try:
        # Process all ETL9G files using the generator
        total_processed = process_etl9g_files(image_files, env, index, limit)
        print(f"Processed {total_processed} records from all files")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Saving progress...")
        save_metadata(env, total_processed, index)
        return env, total_processed
    except Exception as e:
        print(f"Error processing files: {e}")
        print("Saving progress up to this point...")
        save_metadata(env, total_processed, index)
    
    # Final save of metadata
    save_metadata(env, total_processed, index)
    
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
