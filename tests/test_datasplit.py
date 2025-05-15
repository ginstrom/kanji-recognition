"""
Unit tests for the datasplit.py module.
"""
import os
import json
import pickle
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open, call
from collections import defaultdict

from src.datasplit import (
    parse_args,
    collect_character_statistics,
    assign_splits,
    create_target_lmdbs,
    process_split,
    main
)

@pytest.fixture
def mock_char_to_keys():
    """
    Create a mock character-to-keys mapping for testing.
    """
    # Create a mapping with 3 characters, each with multiple keys
    return {
        '漢': ['漢_0_etl9g'.encode(), '漢_1_etl9g'.encode(), '漢_2_etl9g'.encode(), '漢_3_etl9g'.encode(), '漢_4_etl9g'.encode()],
        '字': ['字_0_etl9g'.encode(), '字_1_etl9g'.encode(), '字_2_etl9g'.encode(), '字_3_etl9g'.encode(), '字_4_etl9g'.encode()],
        '認': ['認_0_etl9g'.encode(), '認_1_etl9g'.encode(), '認_2_etl9g'.encode(), '認_3_etl9g'.encode(), '認_4_etl9g'.encode()]
    }

@pytest.fixture
def mock_split_stats():
    """
    Create mock split statistics for testing.
    """
    return {
        'train': {'count': 9, 'characters': {'漢': 3, '字': 3, '認': 3}},
        'val': {'count': 3, 'characters': {'漢': 1, '字': 1, '認': 1}},
        'test': {'count': 3, 'characters': {'漢': 1, '字': 1, '認': 1}}
    }

@pytest.fixture
def mock_metadata():
    """
    Create mock metadata for testing.
    """
    return {
        'total_records': 15,
        'unique_characters': 3,
        'character_counts': {'漢': 5, '字': 5, '認': 5}
    }

@pytest.fixture
def mock_source_env():
    """
    Create a mock source LMDB environment for testing.
    """
    mock_env = MagicMock()
    mock_txn = MagicMock()
    mock_cursor = MagicMock()
    
    # Set up the cursor to return mock key-value pairs
    mock_cursor.__iter__.return_value = [
        ('漢_0_etl9g'.encode(), b'image_data_1'),
        ('漢_1_etl9g'.encode(), b'image_data_2'),
        ('字_0_etl9g'.encode(), b'image_data_3'),
        (b'__metadata__', json.dumps({'total_records': 3}).encode())
    ]
    
    # Set up the transaction to return the cursor and get values
    mock_txn.cursor.return_value = mock_cursor
    mock_txn.get.side_effect = lambda key: {
        '漢_0_etl9g'.encode(): b'image_data_1',
        '漢_1_etl9g'.encode(): b'image_data_2',
        '字_0_etl9g'.encode(): b'image_data_3',
        b'__metadata__': json.dumps({'total_records': 3, 'character_counts': {'漢': 2, '字': 1}}).encode()
    }.get(key)
    
    # Set up the environment to return the transaction
    mock_env.begin.return_value.__enter__.return_value = mock_txn
    mock_env.info.return_value = {'map_size': 1048576}  # 1MB
    
    return mock_env

class TestParseArgs:
    """Tests for the parse_args function."""
    
    def test_default_args(self):
        """Test parsing with default arguments."""
        with patch('argparse.ArgumentParser.parse_args', 
                  return_value=MagicMock(
                      source='/app/output/prep/kanji.lmdb',
                      output_dir='/app/output/prep',
                      train_ratio=0.8,
                      val_ratio=0.1,
                      random_seed=42
                  )):
            args = parse_args()
            
            assert args.source == '/app/output/prep/kanji.lmdb'
            assert args.output_dir == '/app/output/prep'
            assert args.train_ratio == 0.8
            assert args.val_ratio == 0.1
            assert args.random_seed == 42
    
    def test_custom_args(self):
        """Test parsing with custom arguments."""
        with patch('argparse.ArgumentParser.parse_args', 
                  return_value=MagicMock(
                      source='/custom/path/kanji.lmdb',
                      output_dir='/custom/output',
                      train_ratio=0.7,
                      val_ratio=0.15,
                      random_seed=123
                  )):
            args = parse_args()
            
            assert args.source == '/custom/path/kanji.lmdb'
            assert args.output_dir == '/custom/output'
            assert args.train_ratio == 0.7
            assert args.val_ratio == 0.15
            assert args.random_seed == 123
    
    def test_invalid_ratios(self):
        """Test validation of split ratios."""
        # Test negative train ratio
        with patch('argparse.ArgumentParser.parse_args', 
                  return_value=MagicMock(
                      source='/app/output/prep/kanji.lmdb',
                      output_dir='/app/output/prep',
                      train_ratio=-0.1,
                      val_ratio=0.1,
                      random_seed=42
                  )):
            with patch('argparse.ArgumentParser.error') as mock_error:
                parse_args()
                mock_error.assert_called_once_with("Train and validation ratios must be positive")
        
        # Test negative validation ratio
        with patch('argparse.ArgumentParser.parse_args', 
                  return_value=MagicMock(
                      source='/app/output/prep/kanji.lmdb',
                      output_dir='/app/output/prep',
                      train_ratio=0.8,
                      val_ratio=-0.1,
                      random_seed=42
                  )):
            with patch('argparse.ArgumentParser.error') as mock_error:
                parse_args()
                mock_error.assert_called_once_with("Train and validation ratios must be positive")
        
        # Test sum of ratios >= 1.0
        with patch('argparse.ArgumentParser.parse_args', 
                  return_value=MagicMock(
                      source='/app/output/prep/kanji.lmdb',
                      output_dir='/app/output/prep',
                      train_ratio=0.9,
                      val_ratio=0.2,
                      random_seed=42
                  )):
            with patch('argparse.ArgumentParser.error') as mock_error:
                parse_args()
                mock_error.assert_called_once_with("Sum of train and validation ratios must be less than 1.0")

class TestCollectCharacterStatistics:
    """Tests for the collect_character_statistics function."""
    
    def test_collect_with_metadata(self, mock_source_env):
        """Test collecting character statistics with metadata present."""
        # Call the function
        char_to_keys, total_count, metadata = collect_character_statistics(mock_source_env)
        
        # Check the results
        assert isinstance(char_to_keys, defaultdict)
        assert total_count == 3  # 3 keys excluding __metadata__
        assert metadata is not None
        assert metadata['total_records'] == 3
        assert metadata['character_counts'] == {'漢': 2, '字': 1}
        
        # Check that the character-to-keys mapping is correct
        assert len(char_to_keys) == 2  # 2 unique characters
        assert '漢_0_etl9g'.encode() in char_to_keys['漢']
        assert '漢_1_etl9g'.encode() in char_to_keys['漢']
        assert '字_0_etl9g'.encode() in char_to_keys['字']
    
    def test_collect_without_metadata(self):
        """Test collecting character statistics without metadata."""
        # Create a mock environment without metadata
        mock_env = MagicMock()
        mock_txn = MagicMock()
        mock_cursor = MagicMock()
        
        # Set up the cursor to return mock key-value pairs without metadata
        mock_cursor.__iter__.return_value = [
            ('漢_0_etl9g'.encode(), b'image_data_1'),
            ('漢_1_etl9g'.encode(), b'image_data_2'),
            ('字_0_etl9g'.encode(), b'image_data_3')
        ]
        
        # Set up the transaction to return the cursor and get values
        mock_txn.cursor.return_value = mock_cursor
        mock_txn.get.return_value = None  # No metadata
        
        # Set up the environment to return the transaction
        mock_env.begin.return_value.__enter__.return_value = mock_txn
        
        # Call the function
        char_to_keys, total_count, metadata = collect_character_statistics(mock_env)
        
        # Check the results
        assert isinstance(char_to_keys, defaultdict)
        assert total_count == 3
        assert metadata is None
        
        # Check that the character-to-keys mapping is correct
        assert len(char_to_keys) == 2  # 2 unique characters
        assert '漢_0_etl9g'.encode() in char_to_keys['漢']
        assert '漢_1_etl9g'.encode() in char_to_keys['漢']
        assert '字_0_etl9g'.encode() in char_to_keys['字']
    
    def test_collect_with_etl9g_character_counts(self):
        """Test collecting character statistics with character_counts_etl9g in metadata."""
        # Create a mock environment with character_counts_etl9g in metadata
        mock_env = MagicMock()
        mock_txn = MagicMock()
        mock_cursor = MagicMock()
        
        # Set up the cursor to return mock key-value pairs
        mock_cursor.__iter__.return_value = [
            ('漢_0_etl9g'.encode(), b'image_data_1'),
            ('漢_1_etl9g'.encode(), b'image_data_2'),
            ('字_0_etl9g'.encode(), b'image_data_3'),
            (b'__metadata__', json.dumps({'total_records': 3}).encode())
        ]
        
        # Set up the transaction to return the cursor and get values
        mock_txn.cursor.return_value = mock_cursor
        mock_txn.get.side_effect = lambda key: {
            b'__metadata__': json.dumps({
                'total_records': 3, 
                'unique_characters': 2,
                'sources': ['etl9g'],
                'character_counts_etl9g': {'漢': 2, '字': 1}
            }).encode()
        }.get(key)
        
        # Set up the environment to return the transaction
        mock_env.begin.return_value.__enter__.return_value = mock_txn
        
        # Call the function
        char_to_keys, total_count, metadata = collect_character_statistics(mock_env)
        
        # Check the results
        assert isinstance(char_to_keys, defaultdict)
        assert total_count == 3  # 3 keys excluding __metadata__
        assert metadata is not None
        assert metadata['total_records'] == 3
        assert 'character_counts_etl9g' in metadata
        assert metadata['character_counts_etl9g'] == {'漢': 2, '字': 1}
        
        # Check that the character-to-keys mapping is correct
        assert len(char_to_keys) == 2  # 2 unique characters
        assert '漢_0_etl9g'.encode() in char_to_keys['漢']
        assert '漢_1_etl9g'.encode() in char_to_keys['漢']
        assert '字_0_etl9g'.encode() in char_to_keys['字']
    
    def test_collect_with_invalid_keys(self):
        """Test collecting character statistics with invalid keys."""
        # Create a mock environment with some invalid keys
        mock_env = MagicMock()
        mock_txn = MagicMock()
        mock_cursor = MagicMock()
        
        # Set up the cursor to return mock key-value pairs with some invalid keys
        mock_cursor.__iter__.return_value = [
            ('漢_0_etl9g'.encode(), b'image_data_1'),
            (b'invalid_key', b'image_data_2'),  # No character part
            (b'__metadata__', json.dumps({'total_records': 2}).encode())
        ]
        
        # Set up the transaction to return the cursor and get values
        mock_txn.cursor.return_value = mock_cursor
        mock_txn.get.side_effect = lambda key: {
            b'__metadata__': json.dumps({'total_records': 2, 'character_counts': {'漢': 1}}).encode()
        }.get(key)
        
        # Set up the environment to return the transaction
        mock_env.begin.return_value.__enter__.return_value = mock_txn
        
        # Call the function
        char_to_keys, total_count, metadata = collect_character_statistics(mock_env)
        
        # Check the results
        assert isinstance(char_to_keys, defaultdict)
        assert total_count == 2  # 2 keys excluding __metadata__
        assert metadata is not None
        
        # Check that the character-to-keys mapping is correct
        assert len(char_to_keys) == 2  # 2 unique characters (including 'invalid')
        assert '漢_0_etl9g'.encode() in char_to_keys['漢']
        assert b'invalid_key' in char_to_keys['invalid']  # 'invalid' is treated as a character

class TestAssignSplits:
    """Tests for the assign_splits function."""
    
    def test_assign_splits(self, mock_char_to_keys):
        """Test assigning keys to splits."""
        # Call the function with fixed ratios
        train_keys, val_keys, test_keys, split_stats = assign_splits(
            mock_char_to_keys, train_ratio=0.6, val_ratio=0.2, random_seed=42
        )
        
        # Check the results
        assert len(train_keys) == 9  # 60% of 15 keys
        assert len(val_keys) == 3   # 20% of 15 keys
        assert len(test_keys) == 3  # 20% of 15 keys
        
        # Check that all keys are assigned
        all_keys = train_keys + val_keys + test_keys
        assert len(all_keys) == 15
        
        # Check that the split is stratified (each character appears in each split)
        train_chars = set(key.decode().split('_')[0] for key in train_keys)
        val_chars = set(key.decode().split('_')[0] for key in val_keys)
        test_chars = set(key.decode().split('_')[0] for key in test_keys)
        
        assert train_chars == val_chars == test_chars == {'漢', '字', '認'}
        
        # Check the split statistics
        assert split_stats['train']['count'] == 9
        assert split_stats['val']['count'] == 3
        assert split_stats['test']['count'] == 3
        
        # Check that each character appears in each split
        for char in ['漢', '字', '認']:
            assert char in split_stats['train']['characters']
            assert char in split_stats['val']['characters']
            assert char in split_stats['test']['characters']
    
    def test_reproducibility(self, mock_char_to_keys):
        """Test reproducibility with the same random seed."""
        # Call the function twice with the same seed
        train_keys1, val_keys1, test_keys1, _ = assign_splits(
            mock_char_to_keys, train_ratio=0.6, val_ratio=0.2, random_seed=42
        )
        
        train_keys2, val_keys2, test_keys2, _ = assign_splits(
            mock_char_to_keys, train_ratio=0.6, val_ratio=0.2, random_seed=42
        )
        
        # Check that the splits are identical
        assert train_keys1 == train_keys2
        assert val_keys1 == val_keys2
        assert test_keys1 == test_keys2
    
    def test_different_seeds(self, mock_char_to_keys):
        """Test different results with different random seeds."""
        # Call the function twice with different seeds
        train_keys1, val_keys1, test_keys1, _ = assign_splits(
            mock_char_to_keys, train_ratio=0.6, val_ratio=0.2, random_seed=42
        )
        
        train_keys2, val_keys2, test_keys2, _ = assign_splits(
            mock_char_to_keys, train_ratio=0.6, val_ratio=0.2, random_seed=123
        )
        
        # Check that at least one of the splits is different
        # Note: There's a small chance this could fail if the splits happen to be identical by chance
        assert (train_keys1 != train_keys2 or 
                val_keys1 != val_keys2 or 
                test_keys1 != test_keys2)

class TestCreateTargetLmdbs:
    """Tests for the create_target_lmdbs function."""
    
    def test_create_target_lmdbs(self, mock_source_env, mock_split_stats, temp_output_dir):
        """Test creating target LMDB databases."""
        # Create mock train, val, test keys
        train_keys = ['漢_0_etl9g'.encode(), '漢_1_etl9g'.encode(), '字_0_etl9g'.encode()]
        val_keys = ['漢_2_etl9g'.encode()]
        test_keys = ['字_1_etl9g'.encode()]
        
        # Mock metadata
        metadata = {'total_records': 5, 'character_counts': {'漢': 3, '字': 2}}
        
        # Create mock target environments
        mock_train_env = MagicMock()
        mock_val_env = MagicMock()
        mock_test_env = MagicMock()
        
        # Mock lmdb.open to return our mock environments
        with patch('lmdb.open', side_effect=[mock_train_env, mock_val_env, mock_test_env]):
            # Call the function
            train_env, val_env, test_env = create_target_lmdbs(
                mock_source_env, temp_output_dir, train_keys, val_keys, test_keys, mock_split_stats, metadata
            )
            
            # Check that the environments were created
            assert train_env == mock_train_env
            assert val_env == mock_val_env
            assert test_env == mock_test_env
            
            # Check that the output directory was created
            assert os.path.exists(temp_output_dir)
            
            # Check that transactions were created for each environment
            assert mock_train_env.begin.call_count > 0
            assert mock_val_env.begin.call_count > 0
            assert mock_test_env.begin.call_count > 0
    
    def test_error_handling(self, mock_source_env, mock_split_stats, temp_output_dir):
        """Test error handling during target LMDB creation."""
        # Create mock train, val, test keys
        train_keys = ['漢_0_etl9g'.encode()]
        val_keys = ['漢_1_etl9g'.encode()]
        test_keys = ['字_0_etl9g'.encode()]
        
        # Mock metadata
        metadata = {'total_records': 3, 'character_counts': {'漢': 2, '字': 1}}
        
        # Mock source_txn.get to raise an exception for one key
        mock_source_env.begin.return_value.__enter__.return_value.get.side_effect = [
            b'image_data_1',  # First call succeeds
            Exception("Test error"),  # Second call fails
            b'image_data_3'   # Third call succeeds
        ]
        
        # Create mock target environments
        mock_train_env = MagicMock()
        mock_val_env = MagicMock()
        mock_test_env = MagicMock()
        
        # Mock lmdb.open to return our mock environments
        with patch('lmdb.open', side_effect=[mock_train_env, mock_val_env, mock_test_env]):
            # Call the function - it should handle the exception and continue
            train_env, val_env, test_env = create_target_lmdbs(
                mock_source_env, temp_output_dir, train_keys, val_keys, test_keys, mock_split_stats, metadata
            )
            
            # Check that the environments were created despite the error
            assert train_env == mock_train_env
            assert val_env == mock_val_env
            assert test_env == mock_test_env

class TestProcessSplit:
    """Tests for the process_split function."""
    
    def test_process_split_success(self, mock_source_env, mock_split_stats, mock_metadata):
        """Test successful processing of a split."""
        # Create mock keys and target environment
        keys = ['漢_0_etl9g'.encode(), '漢_1_etl9g'.encode(), '字_0_etl9g'.encode()]
        mock_target_env = MagicMock()
        mock_target_txn = MagicMock()
        mock_target_env.begin.return_value.__enter__.return_value = mock_target_txn
        
        # Call the function
        process_split('train', mock_source_env, mock_target_env, keys, mock_split_stats, mock_metadata)
        
        # Check that data was retrieved from source and written to target
        assert mock_source_env.begin.call_count >= len(keys)
        assert mock_target_env.begin.call_count >= len(keys) + 1  # +1 for metadata
        
        # Check that the correct number of keys were written
        assert mock_target_txn.put.call_count >= len(keys) + 1  # +1 for metadata
        
        # Check that metadata was written
        metadata_calls = [call for call in mock_target_txn.put.call_args_list if call[0][0] == b'__metadata__']
        assert len(metadata_calls) == 1
        
        # Verify metadata content
        metadata_json = json.loads(metadata_calls[0][0][1].decode())
        assert metadata_json['split'] == 'train'
        assert metadata_json['num_samples'] == len(keys)
        assert metadata_json['character_counts'] == dict(mock_split_stats['train']['characters'])
        assert metadata_json['original_metadata'] == mock_metadata
    
    def test_process_split_data_error(self, mock_split_stats, mock_metadata):
        """Test error handling when retrieving data from source LMDB."""
        # Create mock keys and environments
        keys = ['漢_0_etl9g'.encode(), '漢_1_etl9g'.encode()]
        
        # Create mock source environment that raises an exception
        mock_source_env = MagicMock()
        mock_source_txn = MagicMock()
        mock_source_txn.get.side_effect = Exception("Test error")
        mock_source_env.begin.return_value.__enter__.return_value = mock_source_txn
        
        # Create mock target environment
        mock_target_env = MagicMock()
        mock_target_txn = MagicMock()
        mock_target_env.begin.return_value.__enter__.return_value = mock_target_txn
        
        # Call the function - it should handle the exception and continue
        process_split('val', mock_source_env, mock_target_env, keys, mock_split_stats, mock_metadata)
        
        # Check that metadata was still written despite the error
        metadata_calls = [call for call in mock_target_txn.put.call_args_list if call[0][0] == b'__metadata__']
        assert len(metadata_calls) == 1
        
        # Verify metadata content
        metadata_json = json.loads(metadata_calls[0][0][1].decode())
        assert metadata_json['split'] == 'val'
        assert metadata_json['num_samples'] == len(keys)
    
    def test_process_split_metadata_error(self, mock_source_env, mock_split_stats, mock_metadata):
        """Test error handling when storing metadata fails."""
        # Create mock keys and target environment
        keys = ['漢_0_etl9g'.encode(), '字_0_etl9g'.encode()]
        
        # Create mock target environment that raises an exception when storing metadata
        mock_target_env = MagicMock()
        mock_target_txn = MagicMock()
        
        # Set up the transaction to raise an exception only for metadata
        def mock_put_side_effect(*args, **kwargs):
            if args[0] == b'__metadata__':
                raise Exception("Metadata error")
        
        mock_target_txn.put.side_effect = mock_put_side_effect
        mock_target_env.begin.return_value.__enter__.return_value = mock_target_txn
        
        # Call the function - it should handle the exception and continue
        process_split('test', mock_source_env, mock_target_env, keys, mock_split_stats, mock_metadata)
        
        # Check that the function attempted to process the keys
        assert mock_source_env.begin.call_count > 0
        assert mock_target_env.begin.call_count > 0
    
    def test_process_split_empty_keys(self, mock_source_env, mock_split_stats, mock_metadata):
        """Test processing a split with no keys."""
        # Create empty keys list and target environment
        keys = []
        mock_target_env = MagicMock()
        mock_target_txn = MagicMock()
        mock_target_env.begin.return_value.__enter__.return_value = mock_target_txn
        
        # Call the function
        process_split('val', mock_source_env, mock_target_env, keys, mock_split_stats, mock_metadata)
        
        # Check that only metadata was written
        assert mock_target_txn.put.call_count == 1
        
        # Verify metadata content
        metadata_json = json.loads(mock_target_txn.put.call_args[0][1].decode())
        assert metadata_json['split'] == 'val'
        assert metadata_json['num_samples'] == 0

class TestMain:
    """Tests for the main function."""
    
    @patch('src.datasplit.parse_args')
    @patch('src.datasplit.lmdb.open')
    @patch('src.datasplit.collect_character_statistics')
    @patch('src.datasplit.assign_splits')
    @patch('src.datasplit.create_target_lmdbs')
    def test_main_success(self, mock_create, mock_assign, mock_collect, mock_open, mock_parse, temp_output_dir):
        """Test successful execution of the main function."""
        # Set up mocks
        mock_args = MagicMock(
            source='/app/output/prep/kanji.lmdb',
            output_dir=temp_output_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            random_seed=42
        )
        mock_parse.return_value = mock_args
        
        mock_env = MagicMock()
        mock_open.return_value = mock_env
        
        mock_char_to_keys = {'漢': ['漢_0_etl9g'.encode()], '字': ['字_0_etl9g'.encode()]}
        mock_metadata = {'total_records': 2}
        mock_collect.return_value = (mock_char_to_keys, 2, mock_metadata)
        
        mock_train_keys = ['漢_0_etl9g'.encode()]
        mock_val_keys = []
        mock_test_keys = ['字_0_etl9g'.encode()]
        mock_split_stats = {
            'train': {'count': 1, 'characters': {'漢': 1}},
            'val': {'count': 0, 'characters': {}},
            'test': {'count': 1, 'characters': {'字': 1}}
        }
        mock_assign.return_value = (mock_train_keys, mock_val_keys, mock_test_keys, mock_split_stats)
        
        mock_train_env = MagicMock()
        mock_val_env = MagicMock()
        mock_test_env = MagicMock()
        mock_create.return_value = (mock_train_env, mock_val_env, mock_test_env)
        
        # Call the function
        main()
        
        # Check that the mocks were called correctly
        mock_parse.assert_called_once()
        mock_open.assert_called_once_with(mock_args.source, readonly=True)
        mock_collect.assert_called_once_with(mock_env)
        mock_assign.assert_called_once_with(
            mock_char_to_keys, mock_args.train_ratio, mock_args.val_ratio, mock_args.random_seed
        )
        mock_create.assert_called_once_with(
            mock_env, mock_args.output_dir, mock_train_keys, mock_val_keys, mock_test_keys, mock_split_stats, mock_metadata
        )
        
        # Check that the environments were closed
        mock_train_env.close.assert_called_once()
        mock_val_env.close.assert_called_once()
        mock_test_env.close.assert_called_once()
        mock_env.close.assert_called_once()
    
    @patch('src.datasplit.parse_args')
    @patch('src.datasplit.lmdb.open')
    def test_main_error_opening_source(self, mock_open, mock_parse, temp_output_dir):
        """Test error handling when opening source LMDB."""
        # Set up mocks
        mock_args = MagicMock(
            source='/app/output/prep/kanji.lmdb',
            output_dir=temp_output_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            random_seed=42
        )
        mock_parse.return_value = mock_args
        
        # Make lmdb.open raise an exception
        mock_open.side_effect = Exception("Test error")
        
        # Call the function - it should handle the exception
        main()
        
        # Check that the mocks were called correctly
        mock_parse.assert_called_once()
        mock_open.assert_called_once_with(mock_args.source, readonly=True)
