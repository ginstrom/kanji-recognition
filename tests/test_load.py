"""
Unit tests for the load.py module.
"""
# import os # Unused F401
import json
import pickle
import pytest
# import numpy as np # Unused F401
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
from torchvision import transforms

from src.load import (
    KanjiLMDBDataset,
    get_transforms,
    create_dataloader,
    get_dataloaders,
    get_class_weights,
    get_num_classes
)

@pytest.fixture
def mock_lmdb_env():
    """Create a mock LMDB environment for testing."""
    mock_env = MagicMock()
    mock_txn = MagicMock()
    
    # Mock metadata
    metadata = {
        'num_samples': 5,
        'split': 'train',
        'character_counts': {'漢': 2, '字': 2, '認': 1}
    }
    
    # Mock sample data
    image = Image.new('L', (128, 128), color=255)
    image_bytes = image.tobytes()
    
    # Set up the transaction to return metadata and sample data
    mock_txn.get.side_effect = lambda key: {
        b'__metadata__': json.dumps(metadata).encode(),
        b'0': pickle.dumps((image_bytes, '漢'.encode())),
        b'1': pickle.dumps((image_bytes, '漢'.encode())),
        b'2': pickle.dumps((image_bytes, '字'.encode())),
        b'3': pickle.dumps((image_bytes, '字'.encode())),
        b'4': pickle.dumps((image_bytes, '認'.encode()))
    }.get(key)
    
    # Set up the environment to return the transaction
    mock_env.begin.return_value.__enter__.return_value = mock_txn
    
    return mock_env

@pytest.fixture
def mock_dataset(mock_lmdb_env):
    """Create a mock KanjiLMDBDataset for testing."""
    with patch('os.path.exists', return_value=True), \
         patch('lmdb.open', return_value=mock_lmdb_env):
        dataset = KanjiLMDBDataset(db_path='/mock/path/to/lmdb')
        yield dataset
        dataset.close()

class TestKanjiLMDBDataset:
    """Tests for the KanjiLMDBDataset class."""
    
    def test_init(self, mock_lmdb_env):
        """Test initialization of the dataset."""
        with patch('os.path.exists', return_value=True), \
             patch('lmdb.open', return_value=mock_lmdb_env):
            dataset = KanjiLMDBDataset(db_path='/mock/path/to/lmdb')
            
            assert dataset.db_path == '/mock/path/to/lmdb'
            assert dataset.transform is None
            assert dataset.target_transform is None
            assert dataset.length == 5
            assert dataset.metadata['split'] == 'train'
            assert dataset.metadata['character_counts'] == {'漢': 2, '字': 2, '認': 1}
            
            dataset.close()
    
    def test_len(self, mock_dataset):
        """Test __len__ method."""
        assert len(mock_dataset) == 5
    
    def test_getitem(self, mock_dataset):
        """Test __getitem__ method."""
        image, label = mock_dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (1, 128, 128)
        assert label == '漢'
    
    def test_get_character_distribution(self, mock_dataset):
        """Test get_character_distribution method."""
        char_dist = mock_dataset.get_character_distribution()
        
        assert char_dist == {'漢': 2, '字': 2, '認': 1}

class TestTransforms:
    """Tests for the transform functions."""
    
    def test_get_transforms_train(self):
        """Test get_transforms for training."""
        transform = get_transforms(split='train', augment=True)
        
        assert isinstance(transform, transforms.Compose)
        transform_str = str(transform)
        assert 'RandomRotation' in transform_str
        assert 'ToTensor' in transform_str
    
    def test_get_transforms_val(self):
        """Test get_transforms for validation."""
        transform = get_transforms(split='val', augment=True)
        
        assert isinstance(transform, transforms.Compose)
        transform_str = str(transform)
        assert 'RandomRotation' not in transform_str
        assert 'ToTensor' in transform_str

class TestDataLoader:
    """Tests for the DataLoader functions."""
    
    def test_create_dataloader(self, mock_lmdb_env):
        """Test create_dataloader function."""
        with patch('os.path.exists', return_value=True), \
             patch('lmdb.open', return_value=mock_lmdb_env):
            dataloader, dataset = create_dataloader(
                db_path='/mock/path/to/lmdb',
                batch_size=2,
                split='train',
                augment=True,
                num_workers=0
            )
            
            assert isinstance(dataloader, torch.utils.data.DataLoader)
            assert isinstance(dataset, KanjiLMDBDataset)
            assert dataloader.batch_size == 2
            
            dataset.close()
    
    def test_get_dataloaders(self, mock_lmdb_env):
        """Test get_dataloaders function."""
        with patch('os.path.exists', return_value=True), \
             patch('lmdb.open', return_value=mock_lmdb_env), \
             patch('os.path.join', return_value='/mock/path/to/lmdb'):
            dataloaders = get_dataloaders(
                data_dir='/mock/data/dir',
                batch_size=2,
                augment=True,
                num_workers=0
            )
            
            assert isinstance(dataloaders, dict)
            
            for split, (loader, dataset) in dataloaders.items():
                dataset.close()

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_class_weights(self, mock_dataset):
        """Test get_class_weights function."""
        weights = get_class_weights(mock_dataset)
        
        assert isinstance(weights, dict)
        assert '漢' in weights
        assert '字' in weights
        assert '認' in weights
    
    def test_get_num_classes(self, mock_dataset):
        """Test get_num_classes function."""
        num_classes = get_num_classes(mock_dataset)
        
        assert num_classes == 3  # Three unique characters
