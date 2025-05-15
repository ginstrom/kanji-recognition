"""
Kanji LMDB Dataset Loader
This module provides a PyTorch Dataset implementation for loading kanji data from LMDB databases.
It handles the deserialization of image and label data, supports transformations for data augmentation,
and provides utility functions for creating DataLoaders for training, validation, and testing.
"""
import os
import json
import pickle
import logging
from typing import Tuple, Dict, List, Optional, Union, Callable, Any

import lmdb
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KanjiLMDBDataset(Dataset):
    """
    PyTorch Dataset for loading kanji data from LMDB databases.
    
    This dataset loads image and label data from LMDB databases created by the datasplit.py script.
    It supports transformations for data augmentation.
    
    Attributes:
        db_path (str): Path to the LMDB database
        transform (callable, optional): Optional transform to be applied to the images
        target_transform (callable, optional): Optional transform to be applied to the labels
        env (lmdb.Environment): LMDB environment
        length (int): Number of samples in the dataset
        metadata (dict): Metadata from the LMDB database
    """
    
    def __init__(self, 
                 db_path: str, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize the KanjiLMDBDataset.
        
        Args:
            db_path (str): Path to the LMDB database
            transform (callable, optional): Optional transform to be applied to the images
            target_transform (callable, optional): Optional transform to be applied to the labels
        
        Raises:
            FileNotFoundError: If the LMDB database does not exist
            RuntimeError: If the LMDB database cannot be opened or metadata cannot be read
        """
        self.db_path = db_path
        self.transform = transform
        self.target_transform = target_transform
        
        # Check if the database exists
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"LMDB database not found at {db_path}")
        
        try:
            # Open the LMDB environment
            self.env = lmdb.open(
                db_path, 
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            
            # Read metadata
            with self.env.begin() as txn:
                metadata_bytes = txn.get(b'__metadata__')
                if metadata_bytes is None:
                    raise RuntimeError(f"No metadata found in LMDB database at {db_path}")
                
                self.metadata = json.loads(metadata_bytes.decode())
                self.length = self.metadata.get('num_samples', 0)
                
                logger.info(f"Loaded dataset from {db_path} with {self.length} samples")
                logger.info(f"Dataset split: {self.metadata.get('split', 'unknown')}")
                
                # Log character distribution if available
                char_counts = self.metadata.get('character_counts', {})
                if char_counts:
                    logger.info(f"Dataset contains {len(char_counts)} unique characters")
        
        except lmdb.Error as e:
            raise RuntimeError(f"Error opening LMDB database at {db_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error initializing dataset from {db_path}: {e}")
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to get
            
        Returns:
            tuple: (image, label) where image is a tensor and label is a string
            
        Raises:
            IndexError: If the index is out of range
            RuntimeError: If the sample cannot be loaded or processed
        """
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset with {self.length} samples")
        
        try:
            # Get the sample from the LMDB database
            with self.env.begin() as txn:
                key = str(idx).encode()
                serialized = txn.get(key)
                
                if serialized is None:
                    raise IndexError(f"Sample with key {key} not found in LMDB database")
                
                # Deserialize the sample
                image_bytes, label_bytes = pickle.loads(serialized)
            
            # Convert image bytes to PIL Image
            # The images are stored as 128x128 grayscale images
            image = Image.frombytes('L', (128, 128), image_bytes)
            
            # Convert label bytes to string
            label = label_bytes.decode()
            
            # Apply transformations if any
            if self.transform:
                image = self.transform(image)
            else:
                # Convert PIL Image to tensor (default transformation)
                image = transforms.ToTensor()(image)
            
            # Apply target transformation if any
            if self.target_transform:
                label = self.target_transform(label)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading sample at index {idx}: {e}")
            raise RuntimeError(f"Error loading sample at index {idx}: {e}")
    
    def get_character_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of characters in the dataset.
        
        Returns:
            dict: Dictionary mapping characters to their counts
        """
        return self.metadata.get('character_counts', {})
    
    def close(self):
        """
        Close the LMDB environment.
        
        This method should be called when the dataset is no longer needed
        to release resources.
        """
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
            self.env = None
    
    def __enter__(self):
        """
        Enter the context manager.
        
        Returns:
            KanjiLMDBDataset: The dataset instance
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        self.close()


def get_transforms(split: str = 'train', 
                   augment: bool = True,
                   img_size: Tuple[int, int] = (128, 128)) -> transforms.Compose:
    """
    Get standard transformations for the specified split.
    
    Args:
        split (str, optional): Dataset split ('train', 'val', or 'test'). Defaults to 'train'.
        augment (bool, optional): Whether to apply data augmentation. Defaults to True.
        img_size (tuple, optional): Target image size. Defaults to (128, 128).
        
    Returns:
        torchvision.transforms.Compose: Composed transformations
    """
    # Base transformations for all splits
    base_transforms = [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ]
    
    # Apply data augmentation only for training
    if split == 'train' and augment:
        train_transforms = [
            transforms.RandomRotation(5),  # Slight rotation
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.05, 0.05),  # Small translations
                scale=(0.95, 1.05),  # Small scaling
                fill=0  # Fill with black
            ),
            # Add slight noise to simulate different writing styles
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 0.5))
            ], p=0.3),
            *base_transforms
        ]
        return transforms.Compose(train_transforms)
    
    # For validation and test, just use base transformations
    return transforms.Compose(base_transforms)


def create_dataloader(db_path: str, 
                      batch_size: int = 32, 
                      split: str = 'train',
                      augment: bool = True,
                      num_workers: int = 4,
                      shuffle: bool = None,
                      pin_memory: bool = True,
                      **kwargs) -> Tuple[DataLoader, KanjiLMDBDataset]:
    """
    Create a DataLoader for the specified LMDB database.
    
    Args:
        db_path (str): Path to the LMDB database
        batch_size (int, optional): Batch size. Defaults to 32.
        split (str, optional): Dataset split ('train', 'val', or 'test'). Defaults to 'train'.
        augment (bool, optional): Whether to apply data augmentation. Defaults to True.
        num_workers (int, optional): Number of worker processes. Defaults to 4.
        shuffle (bool, optional): Whether to shuffle the data. If None, will be True for 'train' and False otherwise.
        pin_memory (bool, optional): Whether to pin memory in GPU training. Defaults to True.
        **kwargs: Additional arguments to pass to the DataLoader
        
    Returns:
        tuple: (DataLoader, KanjiLMDBDataset) The created DataLoader and Dataset
    """
    # Set default shuffle based on split if not specified
    if shuffle is None:
        shuffle = (split == 'train')
    
    # Get appropriate transforms
    transform = get_transforms(split, augment)
    
    # Create dataset
    dataset = KanjiLMDBDataset(
        db_path=db_path,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )
    
    return dataloader, dataset


def get_dataloaders(data_dir: str = '/app/output/prep',
                    batch_size: int = 32,
                    augment: bool = True,
                    num_workers: int = 4,
                    **kwargs) -> Dict[str, Tuple[DataLoader, KanjiLMDBDataset]]:
    """
    Get DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir (str, optional): Directory containing the LMDB databases. Defaults to '/app/output/prep'.
        batch_size (int, optional): Batch size. Defaults to 32.
        augment (bool, optional): Whether to apply data augmentation. Defaults to True.
        num_workers (int, optional): Number of worker processes. Defaults to 4.
        **kwargs: Additional arguments to pass to the DataLoader
        
    Returns:
        dict: Dictionary mapping split names to (DataLoader, Dataset) tuples
    """
    train_path = os.path.join(data_dir, 'kanji.train.lmdb')
    val_path = os.path.join(data_dir, 'kanji.val.lmdb')
    test_path = os.path.join(data_dir, 'kanji.test.lmdb')
    
    # Check if the databases exist
    dataloaders = {}
    
    if os.path.exists(train_path):
        train_loader, train_dataset = create_dataloader(
            train_path, 
            batch_size=batch_size, 
            split='train',
            augment=augment,
            num_workers=num_workers,
            **kwargs
        )
        dataloaders['train'] = (train_loader, train_dataset)
    else:
        logger.warning(f"Train dataset not found at {train_path}")
    
    if os.path.exists(val_path):
        val_loader, val_dataset = create_dataloader(
            val_path, 
            batch_size=batch_size, 
            split='val',
            augment=False,  # No augmentation for validation
            num_workers=num_workers,
            **kwargs
        )
        dataloaders['val'] = (val_loader, val_dataset)
    else:
        logger.warning(f"Validation dataset not found at {val_path}")
    
    if os.path.exists(test_path):
        test_loader, test_dataset = create_dataloader(
            test_path, 
            batch_size=batch_size, 
            split='test',
            augment=False,  # No augmentation for test
            num_workers=num_workers,
            **kwargs
        )
        dataloaders['test'] = (test_loader, test_dataset)
    else:
        logger.warning(f"Test dataset not found at {test_path}")
    
    return dataloaders


def get_class_weights(dataset: KanjiLMDBDataset) -> Dict[str, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        dataset (KanjiLMDBDataset): The dataset to calculate weights for
        
    Returns:
        Dict[str, float]: Dictionary mapping characters to weights, or None if no distribution is available
    """
    # Get character distribution
    char_counts = dataset.get_character_distribution()
    
    if not char_counts:
        logger.warning("No character distribution found in dataset metadata. Using equal weights.")
        return None
    
    # Get unique characters and their counts
    chars = list(char_counts.keys())
    counts = [char_counts[char] for char in chars]
    
    # Calculate weights (inverse of frequency)
    total = sum(counts)
    weights = {char: total / (len(chars) * count) for char, count in zip(chars, counts)}
    
    return weights


def get_num_classes(dataset: KanjiLMDBDataset) -> int:
    """
    Get the number of unique classes (characters) in the dataset.
    
    Args:
        dataset (KanjiLMDBDataset): The dataset to get classes from
        
    Returns:
        int: Number of unique classes
    """
    char_counts = dataset.get_character_distribution()
    return len(char_counts)
