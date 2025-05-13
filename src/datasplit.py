import os
import numpy as np
import pickle
import random
from PIL import Image
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def split_dataset(kanji_dicts, train_ratio=0.8, val_ratio=0.1, 
                  stratify_by_char=True, random_seed=42):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        kanji_dicts: List of dictionaries, each with 'character', 'original', 
                    'cropped', and 'twoBit' fields
        train_ratio: Fraction of data to use for training (default: 0.8)
        val_ratio: Fraction of data to use for validation (default: 0.1)
                  (remaining fraction will be used for testing)
        stratify_by_char: Whether to stratify the split by character class (default: True)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing 'train', 'val', and 'test' lists of indices
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    num_samples = len(kanji_dicts)
    test_ratio = 1.0 - train_ratio - val_ratio
    
    indices = list(range(num_samples))
    
    # If stratifying by character, ensure each split has a proportional representation of each class
    if stratify_by_char:
        # Group indices by character using defaultdict(list)
        char_to_indices = defaultdict(list)
        for i, item in enumerate(kanji_dicts):
            char = item['character']
            char_to_indices[char].append(i)
        
        # Check class distribution 
        class_counts = {char: len(indices) for char, indices in char_to_indices.items()}
        print(f"Dataset contains {len(class_counts)} unique characters")
        print(f"Min samples per class: {min(class_counts.values())}")
        print(f"Max samples per class: {max(class_counts.values())}")
        print(f"Avg samples per class: {sum(class_counts.values()) / len(class_counts):.1f}")
        
        # Create stratified splits
        train_indices = []
        val_indices = []
        test_indices = []
        
        for char, char_indices in char_to_indices.items():
            # Shuffle the indices for this character
            random.shuffle(char_indices)
            
            # Calculate split sizes for this character
            n_train = int(len(char_indices) * train_ratio)
            n_val = int(len(char_indices) * val_ratio)
            
            # Split the indices
            train_indices.extend(char_indices[:n_train])
            val_indices.extend(char_indices[n_train:n_train + n_val])
            test_indices.extend(char_indices[n_train + n_val:])
    else:
        # Simple random split
        random.shuffle(indices)
        
        # Calculate split sizes
        n_train = int(num_samples * train_ratio)
        n_val = int(num_samples * val_ratio)
        
        # Split the indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
    
    # Verify splits
    print(f"Train set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")
    
    return {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }