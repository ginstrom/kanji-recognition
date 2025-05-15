"""
# Dataset Splitting Script
This script splits the kanji LMDB database into train, validation, and test sets.
It creates three separate LMDB databases with configurable split ratios.
"""
import os
import argparse
import json
import pickle
import random
import numpy as np
import lmdb
from collections import defaultdict
from tqdm import tqdm

def parse_args():
    """
    Parse command-line arguments for dataset splitting.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Split kanji LMDB database into train, validation, and test sets')
    
    parser.add_argument('--source', type=str, default='/app/output/prep/kanji.lmdb',
                        help='Path to source LMDB database (default: /app/output/prep/kanji.lmdb)')
    
    parser.add_argument('--output-dir', type=str, default='/app/output/prep',
                        help='Directory to store output LMDB databases (default: /app/output/prep)')
    
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Fraction of data to use for training (default: 0.8)')
    
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Fraction of data to use for validation (default: 0.1)')
    
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate split ratios
    if args.train_ratio <= 0 or args.val_ratio <= 0:
        parser.error("Train and validation ratios must be positive")
    
    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("Sum of train and validation ratios must be less than 1.0")
    
    return args

def collect_character_statistics(env_source):
    """
    Collect character statistics from the source LMDB database.
    
    Args:
        env_source: Source LMDB environment
        
    Returns:
        tuple: (character-to-keys mapping, total count, metadata)
    """
    print("Collecting character statistics from source LMDB...")
    char_to_keys = defaultdict(list)
    total_count = 0
    metadata = None
    
    # Read metadata if available
    with env_source.begin() as txn:
        metadata_bytes = txn.get(b'__metadata__')
        if metadata_bytes:
            metadata = json.loads(metadata_bytes.decode())
            # Check for different character count keys based on the source
            if 'character_counts' in metadata:
                char_counts_key = 'character_counts'
            elif 'character_counts_etl9g' in metadata:
                char_counts_key = 'character_counts_etl9g'
            else:
                # If no character counts found, create an empty dict
                metadata['character_counts'] = {}
                char_counts_key = 'character_counts'
                
            print(f"Found metadata: {len(metadata[char_counts_key])} unique characters, "
                  f"{metadata['total_records']} total records")
    
    # Collect keys by character
    with env_source.begin() as txn:
        cursor = txn.cursor()
        for key, _ in tqdm(cursor, desc="Scanning source LMDB"):
            # Skip metadata key
            if key == b'__metadata__':
                continue
            
            # Parse key to extract character
            # Expected format: character_index_source
            key_str = key.decode()
            parts = key_str.split('_')
            if len(parts) >= 1:
                character = parts[0]
                char_to_keys[character].append(key)
                total_count += 1
    
    print(f"Found {len(char_to_keys)} unique characters, {total_count} total records")
    
    return char_to_keys, total_count, metadata

def assign_splits(char_to_keys, train_ratio, val_ratio, random_seed=42):
    """
    Assign each key to a split (train, val, test) using stratified sampling.
    
    Args:
        char_to_keys: Dictionary mapping characters to lists of keys
        train_ratio: Fraction of data to use for training
        val_ratio: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_keys, val_keys, test_keys, split_stats)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    train_keys = []
    val_keys = []
    test_keys = []
    
    # Calculate test ratio
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # Track statistics for each split
    split_stats = {
        'train': {'count': 0, 'characters': defaultdict(int)},
        'val': {'count': 0, 'characters': defaultdict(int)},
        'test': {'count': 0, 'characters': defaultdict(int)}
    }
    
    print("Performing stratified split by character...")
    
    # Check class distribution 
    class_counts = {char: len(keys) for char, keys in char_to_keys.items()}
    print(f"Dataset contains {len(class_counts)} unique characters")
    print(f"Min samples per class: {min(class_counts.values())}")
    print(f"Max samples per class: {max(class_counts.values())}")
    print(f"Avg samples per class: {sum(class_counts.values()) / len(class_counts):.1f}")
    
    # For each character, split its keys into train, val, test
    for char, keys in tqdm(char_to_keys.items(), desc="Assigning splits"):
        # Shuffle the keys for this character
        shuffled_keys = keys.copy()
        random.shuffle(shuffled_keys)
        
        # Calculate split sizes for this character
        n_train = int(len(shuffled_keys) * train_ratio)
        n_val = int(len(shuffled_keys) * val_ratio)
        
        # Assign keys to splits
        char_train_keys = shuffled_keys[:n_train]
        char_val_keys = shuffled_keys[n_train:n_train + n_val]
        char_test_keys = shuffled_keys[n_train + n_val:]
        
        # Add to overall split lists
        train_keys.extend(char_train_keys)
        val_keys.extend(char_val_keys)
        test_keys.extend(char_test_keys)
        
        # Update statistics
        split_stats['train']['count'] += len(char_train_keys)
        split_stats['train']['characters'][char] = len(char_train_keys)
        
        split_stats['val']['count'] += len(char_val_keys)
        split_stats['val']['characters'][char] = len(char_val_keys)
        
        split_stats['test']['count'] += len(char_test_keys)
        split_stats['test']['characters'][char] = len(char_test_keys)
    
    # Print split statistics
    print(f"Train set: {len(train_keys)} samples")
    print(f"Validation set: {len(val_keys)} samples")
    print(f"Test set: {len(test_keys)} samples")
    
    return train_keys, val_keys, test_keys, split_stats

def process_split(split_name, env_source, env_target, keys, split_stats, metadata=None):
    """
    Process a single data split (train, validation, or test).
    
    Args:
        split_name: Name of the split ('train', 'val', or 'test')
        env_source: Source LMDB environment
        env_target: Target LMDB environment
        keys: List of keys for this split
        split_stats: Statistics about all splits
        metadata: Original metadata from source LMDB
    """
    print(f"Processing {split_name} split...")
    for i, key in enumerate(tqdm(keys, desc=f"Writing {split_name} LMDB")):
        try:
            # Get data from source in a short-lived read transaction
            with env_source.begin() as source_txn:
                value = source_txn.get(key)
                if value is None:
                    print(f"Warning: Key {key} not found in source LMDB")
                    continue
                
                # Extract character from key
                char = key.decode().split('_')[0]
                
                # Store as (image, label) tuple for PyTorch Dataset compatibility
                data_tuple = (value, char.encode())
                serialized = pickle.dumps(data_tuple)
            
            # Write to target in a short-lived write transaction
            with env_target.begin(write=True) as target_txn:
                # Use sequential integer keys
                target_txn.put(str(i).encode(), serialized)
        except Exception as e:
            print(f"Error processing {split_name} key {key}: {e}")
    
    # Store metadata in a separate transaction
    try:
        with env_target.begin(write=True) as target_txn:
            split_metadata = {
                'num_samples': len(keys),
                'split': split_name,
                'character_counts': dict(split_stats[split_name]['characters']),
                'original_metadata': metadata
            }
            target_txn.put(b'__metadata__', json.dumps(split_metadata).encode())
    except Exception as e:
        print(f"Error storing {split_name} metadata: {e}")

def create_target_lmdbs(env_source, output_dir, train_keys, val_keys, test_keys, split_stats, metadata=None):
    """
    Create target LMDB databases and transfer data.
    
    Args:
        env_source: Source LMDB environment
        output_dir: Directory to store output LMDB databases
        train_keys: List of keys for training set
        val_keys: List of keys for validation set
        test_keys: List of keys for test set
        split_stats: Statistics about the splits
        metadata: Original metadata from source LMDB
        
    Returns:
        tuple: (train_env, val_env, test_env)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create target LMDB environments
    train_path = os.path.join(output_dir, 'kanji.train.lmdb')
    val_path = os.path.join(output_dir, 'kanji.val.lmdb')
    test_path = os.path.join(output_dir, 'kanji.test.lmdb')
    
    # Estimate map size based on source LMDB size and split ratios
    source_map_size = env_source.info()['map_size']
    total_keys = len(train_keys) + len(val_keys) + len(test_keys)
    
    train_map_size = int(source_map_size * (len(train_keys) / total_keys) * 1.2) if total_keys > 0 else source_map_size
    val_map_size = int(source_map_size * (len(val_keys) / total_keys) * 1.2) if total_keys > 0 else source_map_size
    test_map_size = int(source_map_size * (len(test_keys) / total_keys) * 1.2) if total_keys > 0 else source_map_size
    
    # Ensure minimum map size
    min_map_size = 10 * 1024 * 1024  # 10 MB
    train_map_size = max(train_map_size, min_map_size)
    val_map_size = max(val_map_size, min_map_size)
    test_map_size = max(test_map_size, min_map_size)
    
    # Open target LMDB environments
    train_env = lmdb.open(train_path, map_size=train_map_size)
    val_env = lmdb.open(val_path, map_size=val_map_size)
    test_env = lmdb.open(test_path, map_size=test_map_size)
    
    # Process each split
    process_split('train', env_source, train_env, train_keys, split_stats, metadata)
    process_split('val', env_source, val_env, val_keys, split_stats, metadata)
    process_split('test', env_source, test_env, test_keys, split_stats, metadata)
    
    print(f"Created train LMDB at {train_path}")
    print(f"Created validation LMDB at {val_path}")
    print(f"Created test LMDB at {test_path}")
    
    return train_env, val_env, test_env

def main():
    """
    Main function to split the kanji LMDB database.
    """
    # Parse command-line arguments
    args = parse_args()
    
    print(f"Source LMDB: {args.source}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Test ratio: {1.0 - args.train_ratio - args.val_ratio}")
    print(f"Random seed: {args.random_seed}")
    
    # Open source LMDB
    try:
        env_source = lmdb.open(args.source, readonly=True)
    except Exception as e:
        print(f"Error opening source LMDB: {e}")
        return
    
    try:
        # Collect character statistics
        char_to_keys, total_count, metadata = collect_character_statistics(env_source)
        
        # Assign splits
        train_keys, val_keys, test_keys, split_stats = assign_splits(
            char_to_keys, args.train_ratio, args.val_ratio, args.random_seed
        )
        
        # Create target LMDBs
        train_env, val_env, test_env = create_target_lmdbs(
            env_source, args.output_dir, train_keys, val_keys, test_keys, split_stats, metadata
        )
        
        # Close environments
        train_env.close()
        val_env.close()
        test_env.close()
        
        print("Dataset splitting complete!")
        
    except Exception as e:
        print(f"Error during dataset splitting: {e}")
    finally:
        # Close source environment
        env_source.close()

"""
PyTorch Dataset Implementation Notes:

To load data from these LMDB databases in a PyTorch Dataset, a separate load.py file should be created with a class similar to:

class KanjiLMDBDataset(torch.utils.data.Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, readonly=True, lock=False)
        self.transform = transform
        
        # Get the number of samples from metadata
        with self.env.begin() as txn:
            metadata = json.loads(txn.get(b'__metadata__').decode())
            self.length = metadata['num_samples']
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            # Get serialized (image_bytes, label) tuple
            serialized = txn.get(str(idx).encode())
            if serialized is None:
                raise IndexError(f"Index {idx} out of range")
            
            # Deserialize the tuple
            image_bytes, label_bytes = pickle.loads(serialized)
            
            # Convert image bytes to PIL Image
            image = Image.frombytes('L', (128, 128), image_bytes)
            
            # Convert label bytes to string
            label = label_bytes.decode()
            
            # Apply transformations if any
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    def close(self):
        self.env.close()
"""

if __name__ == "__main__":
    main()
