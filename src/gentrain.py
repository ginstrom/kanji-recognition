import os
import argparse
import json
import lmdb
import pickle
from collections import defaultdict
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge font-generated kanji data into an existing training LMDB database."
    )
    parser.add_argument(
        '--font_db_path', type=str, default='/app/output/prep/kanji_fonts.lmdb',
        help='Path to the source LMDB database containing font-generated kanji images.'
    )
    parser.add_argument(
        '--train_db_path', type=str, default='/app/output/prep/kanji.train.lmdb',
        help='Path to the target training LMDB database to merge data into.'
    )
    return parser.parse_args()

def _read_lmdb_metadata_and_info(db_path, db_name_for_logging="DB"):
    """Reads metadata and env.info() from an LMDB database."""
    metadata = {}
    db_info = {}
    num_samples = 0
    map_size = 1 * 1024 * 1024 * 1024  # Default 1GB
    map_used = 0

    if not os.path.exists(db_path):
        logger.warning(f"{db_name_for_logging} at {db_path} not found for metadata read.")
        return metadata, db_info, num_samples, map_size, map_used

    env_ro = None
    try:
        env_ro = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env_ro.begin() as txn:
            meta_bytes = txn.get(b'__metadata__')
            if meta_bytes:
                metadata = json.loads(meta_bytes.decode())
            else:
                logger.warning(f"No metadata found in {db_name_for_logging} at {db_path}.")
        
        db_info = env_ro.info()
        num_samples = metadata.get('num_samples', metadata.get('total_records', 0)) # handles both train and font DB meta
        map_size = db_info.get('map_size', map_size)
        map_used = db_info.get('map_used', 0)
        
    except lmdb.Error as e:
        logger.error(f"LMDBError reading {db_name_for_logging} {db_path} for info: {e}.")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding metadata from {db_name_for_logging} {db_path}: {e}.")
        metadata = {} # Reset if corrupt
    except Exception as e:
        logger.error(f"Generic error reading {db_name_for_logging} {db_path} for info: {e}.")
    finally:
        if env_ro:
            env_ro.close()
            
    return metadata, db_info, num_samples, map_size, map_used

def _calculate_target_map_size(current_train_map_size, current_train_map_used, 
                               current_train_num_samples, estimated_font_records_to_add):
    """Estimates the new map_size for the training LMDB."""
    avg_item_size_train = 20 * 1024  # Default 20KB per item
    if current_train_num_samples > 0 and current_train_map_used > 0:
        avg_item_size_train = current_train_map_used / current_train_num_samples
    
    estimated_additional_size_needed = estimated_font_records_to_add * avg_item_size_train
    
    target_map_size = int(current_train_map_used + estimated_additional_size_needed * 1.5) # 50% buffer
    target_map_size = max(target_map_size, current_train_map_size) # At least current
    target_map_size = max(target_map_size, 100 * 1024 * 1024)     # Min 100MB
    
    logger.info(f"Original train_db map_size: {current_train_map_size / (1024*1024):.2f} MB, Used: {current_train_map_used / (1024*1024):.2f} MB")
    logger.info(f"Estimated font records to add: {estimated_font_records_to_add}")
    logger.info(f"Avg item size (train): {avg_item_size_train / 1024:.2f} KB")
    logger.info(f"Calculated target map_size for train_db: {target_map_size / (1024*1024):.2f} MB")
    return target_map_size

def _transfer_data_and_update_stats(font_env, train_env, 
                                    current_train_num_samples, initial_train_char_counts,
                                    font_total_items_for_tqdm=None):
    """Iterates through font_db, adds data to train_db, and updates counts."""
    num_font_samples_added = 0
    font_chars_added_summary = defaultdict(int)
    effective_train_char_counts = defaultdict(int, initial_train_char_counts)
    next_train_idx = current_train_num_samples

    with font_env.begin(write=False) as txn_font:
        cursor = txn_font.cursor()
        progress_bar = tqdm(cursor, total=font_total_items_for_tqdm, desc="Merging font data")
        
        for font_key_bytes, font_img_bytes in progress_bar:
            if font_key_bytes == b'__metadata__':
                continue
            try:
                font_key_str = font_key_bytes.decode('utf-8')
                char = font_key_str.split('_')[0] # Assumes format: "{character}_{...}"

                data_tuple = (font_img_bytes, char.encode('utf-8'))
                serialized_data = pickle.dumps(data_tuple)

                with train_env.begin(write=True) as txn_train:
                    train_key = str(next_train_idx).encode('utf-8')
                    txn_train.put(train_key, serialized_data)
                
                effective_train_char_counts[char] += 1
                font_chars_added_summary[char] += 1
                num_font_samples_added += 1
                next_train_idx += 1
            except Exception as e:
                logger.error(f"Error processing font_key {font_key_bytes.decode('utf-8', errors='ignore')}: {e}")
                
    return num_font_samples_added, font_chars_added_summary, effective_train_char_counts, next_train_idx

def _update_train_db_metadata(train_env, train_meta_original, num_font_samples_added, 
                              font_chars_added_summary, font_db_path, 
                              final_train_char_counts, final_num_samples):
    """Updates and writes the metadata to the training LMDB."""
    updated_meta = train_meta_original.copy() # Start with original or empty if none
    
    updated_meta['num_samples'] = final_num_samples
    updated_meta['character_counts'] = dict(final_train_char_counts)
    
    if 'split' not in updated_meta: updated_meta['split'] = 'train'
    if 'original_metadata' not in updated_meta: updated_meta['original_metadata'] = {}

    updated_meta['merged_font_data_stats'] = {
        "source_path": font_db_path,
        "num_samples_added": num_font_samples_added,
        "character_counts_added": dict(font_chars_added_summary)
    }

    with train_env.begin(write=True) as txn_train:
        txn_train.put(b'__metadata__', json.dumps(updated_meta).encode('utf-8'))
    
    logger.info(f"Successfully updated metadata. New total samples: {updated_meta['num_samples']}")
    return updated_meta


def merge_font_data_to_train(font_db_path, train_db_path):
    """
    Merges data from font_db into train_db and updates metadata.
    Orchestrates calls to helper functions.
    """
    if not os.path.exists(font_db_path):
        logger.error(f"Font LMDB database not found: {font_db_path}")
        return
    if not os.path.exists(train_db_path): # train_db can be new/empty, but path should be writable
        logger.info(f"Training LMDB database {train_db_path} not found. Will attempt to create.")
        # Ensure parent directory exists for train_db_path if it's new
        train_db_dir = os.path.dirname(train_db_path)
        if train_db_dir and not os.path.exists(train_db_dir):
            try:
                os.makedirs(train_db_dir, exist_ok=True)
                logger.info(f"Created directory for new train_db: {train_db_dir}")
            except OSError as e:
                logger.error(f"Could not create directory {train_db_dir} for train_db: {e}")
                return


    font_env = None
    train_env = None
    target_map_size_for_error_msg = 0 # For MapFullError message

    try:
        # --- 1. Read Font DB Info ---
        font_meta, _, font_total_records, _, _ = _read_lmdb_metadata_and_info(font_db_path, "FontDB")
        logger.info(f"Font DB ({font_db_path}) metadata: {font_total_records} total records (from metadata).")

        # --- 2. Read Train DB Info ---
        train_meta_initial, _, current_train_num_samples, current_train_map_size, current_train_map_used = \
            _read_lmdb_metadata_and_info(train_db_path, "TrainDB")
        initial_train_char_counts = train_meta_initial.get('character_counts', {})
        logger.info(f"Train DB ({train_db_path}) initial: {current_train_num_samples} samples, "
                    f"MapSize: {current_train_map_size/(1024*1024):.2f}MB, Used: {current_train_map_used/(1024*1024):.2f}MB.")

        # --- 3. Calculate Target Map Size ---
        # If font_total_records is 0 (e.g. no metadata or empty DB), we need to open font_env to count
        estimated_font_records_to_add = font_total_records
        if estimated_font_records_to_add == 0 and os.path.exists(font_db_path):
             temp_font_env = None
             try:
                temp_font_env = lmdb.open(font_db_path, readonly=True, lock=False)
                with temp_font_env.begin() as txn:
                    estimated_font_records_to_add = txn.stat()['entries'] -1 # -1 for metadata
                logger.info(f"Counted {estimated_font_records_to_add} entries in font DB for map size estimation.")
             except Exception as e:
                logger.warning(f"Could not count font DB entries for map size, using default of 500k: {e}")
                estimated_font_records_to_add = 500000 # Fallback
             finally:
                if temp_font_env: temp_font_env.close()
        
        target_map_size = _calculate_target_map_size(
            current_train_map_size, current_train_map_used, 
            current_train_num_samples, estimated_font_records_to_add
        )
        target_map_size_for_error_msg = target_map_size


        # --- 4. Open LMDB environments ---
        font_env = lmdb.open(font_db_path, readonly=True, lock=False, readahead=False, meminit=False)
        train_env = lmdb.open(train_db_path, map_size=target_map_size, writemap=True) # writemap=True for performance potentially

        # --- 5. Transfer Data ---
        logger.info(f"Starting data transfer from {font_db_path} to {train_db_path}...")
        num_added, added_chars_summary, final_char_counts, final_total_samples_count = \
            _transfer_data_and_update_stats(
                font_env, train_env,
                current_train_num_samples, initial_train_char_counts,
                font_total_items_for_tqdm=estimated_font_records_to_add # Use the potentially recounted value
            )
        
        logger.info(f"Added {num_added} samples from {font_db_path}.")

        # --- 6. Update Train DB Metadata ---
        _update_train_db_metadata(
            train_env, train_meta_initial, num_added, added_chars_summary,
            font_db_path, final_char_counts, final_total_samples_count
        )

    except lmdb.MapFullError:
        logger.error(
            f"LMDB MapFullError for {train_db_path}. "
            f"The calculated map_size of {target_map_size_for_error_msg / (1024*1024):.2f} MB was insufficient. "
            "Consider increasing it manually or re-running with a larger base estimate if font DB was empty/no-meta."
        )
    except Exception as e:
        logger.error(f"An error occurred during the merge process: {e}", exc_info=True)
    finally:
        if font_env:
            font_env.close()
        if train_env:
            train_env.close()
        logger.info("LMDB environments closed.")

def main():
    args = parse_args()
    logger.info(f"Starting merge process:")
    logger.info(f"  Font DB (source): {args.font_db_path}")
    logger.info(f"  Train DB (target): {args.train_db_path}")
    
    merge_font_data_to_train(args.font_db_path, args.train_db_path)
    
    logger.info("Merge process finished.")

if __name__ == "__main__":
    main() 