"""
Unit tests for the prepare.py module.
"""
import os
import json
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock, mock_open
from collections import defaultdict

from src.prepare import (
    setup_lmdb,
    process_and_store_kanji,
    save_metadata,
    process_etl9g_files,
    prepare_etl9g_dataset
)

class TestSetupLmdb:
    """Tests for the setup_lmdb function."""
    
    def test_setup_lmdb(self, temp_output_dir):
        """Test setting up an LMDB environment."""
        # Call the function
        env, index = setup_lmdb(temp_output_dir, map_size=10485760)  # 10MB
        
        try:
            # Check that the environment was created
            assert env is not None
            
            # Check that the index is a defaultdict
            assert isinstance(index, defaultdict)
            assert index.default_factory == int
            
            # Check that the database directory was created
            db_path = os.path.join(temp_output_dir, 'kanji.lmdb')
            assert os.path.exists(db_path)
            assert os.path.isdir(db_path)
        finally:
            # Clean up
            env.close()

class TestProcessAndStoreKanji:
    """Tests for the process_and_store_kanji function."""
    
    def test_process_and_store_kanji(self, mock_kanji_dict, mock_lmdb_env):
        """Test processing and storing a kanji dictionary in LMDB."""
        # Add a twoBit field to the mock dictionary
        kanji_dict = {**mock_kanji_dict, 'twoBit': Image.fromarray(np.ones((128, 128), dtype=np.uint8))}
        
        # Create a defaultdict for the index
        index = defaultdict(int)
        
        # Start a transaction
        with mock_lmdb_env.begin(write=True) as txn:
            # Call the function
            key = process_and_store_kanji(kanji_dict, txn, index)
            
            # Check that the key has the expected format
            assert key == f"{kanji_dict['character']}_0_etl9g"
            
            # Check that the index was updated
            assert index[kanji_dict['character']] == 1
            
            # Check that the data was stored in the database
            stored_data = txn.get(key.encode())
            assert stored_data is not None
            
            # The stored data should be the bytes of the twoBit image
            expected_bytes = kanji_dict['twoBit'].tobytes()
            assert stored_data == expected_bytes

class TestSaveMetadata:
    """Tests for the save_metadata function."""
    
    def test_save_metadata_success(self, mock_lmdb_env):
        """Test successful saving of metadata to LMDB."""
        # Create test data
        total_processed = 100
        index = {'漢': 50, '字': 50}
        
        # Call the function
        save_metadata(mock_lmdb_env, total_processed, index)
        
        # Check that the metadata was stored
        with mock_lmdb_env.begin() as txn:
            metadata_bytes = txn.get(b'__metadata__')
            assert metadata_bytes is not None
            
            # Parse the metadata
            metadata = json.loads(metadata_bytes.decode())
            
            # Check the metadata contents
            assert metadata['total_records'] == total_processed
            assert metadata['unique_characters'] == len(index)
            assert metadata['character_counts'] == index
    
    def test_save_metadata_fallback(self, temp_output_dir):
        """Test fallback to file when LMDB save fails."""
        # Create test data
        total_processed = 100
        index = {'漢': 50, '字': 50}
        
        # Create a mock environment that will raise an exception when begin is called
        mock_env = MagicMock()
        mock_env.begin.side_effect = Exception("Test error")
        mock_env.path.return_value = os.path.join(temp_output_dir, 'test.lmdb')
        
        # Mock the open function
        m = mock_open()
        with patch('builtins.open', m):
            # Call the function
            save_metadata(mock_env, total_processed, index)
            
            # Check that the file was opened for writing
            m.assert_called_once_with(os.path.join(os.path.dirname(mock_env.path()), 'metadata.json'), 'w')
            
            # Check that json.dump was called with the expected data
            handle = m()
            handle.write.assert_called()  # json.dump will call write

class TestProcessEtl9gFiles:
    """Tests for the process_etl9g_files function."""
    
    @patch('src.prepare.extract_images')
    def test_process_etl9g_files(self, mock_extract):
        """Test processing ETL9G files and storing in LMDB."""
        # Create mock data
        mock_items = [
            {
                'original': Image.fromarray(np.zeros((128, 128), dtype=np.uint8)),
                'cropped': Image.fromarray(np.zeros((128, 128), dtype=np.uint8)),
                'character': '漢',
                'twoBit': Image.fromarray(np.zeros((128, 128), dtype=np.uint8))
            },
            {
                'original': Image.fromarray(np.zeros((128, 128), dtype=np.uint8)),
                'cropped': Image.fromarray(np.zeros((128, 128), dtype=np.uint8)),
                'character': '字',
                'twoBit': Image.fromarray(np.zeros((128, 128), dtype=np.uint8))
            }
        ]
        
        # Set up the mock to return our test items
        mock_extract.return_value = mock_items
        
        # Create a mock LMDB environment and transaction
        mock_env = MagicMock()
        mock_txn = MagicMock()
        mock_env.begin.return_value.__enter__.return_value = mock_txn
        
        # Create a defaultdict for the index
        index = defaultdict(int)
        
        # Call the function
        count = process_etl9g_files(['file1.bin'], mock_env, index)
        
        # Check the results
        assert count == len(mock_items)
        assert index['漢'] == 1
        assert index['字'] == 1
        
        # Check that put was called on the transaction for each item
        assert mock_txn.put.call_count == len(mock_items)
        
        # The keys should be in the format character_index_etl9g
        # where index is 0 for each unique character (since we're using a fresh index)
        expected_keys = [f"{item['character']}_0_etl9g".encode() for item in mock_items]
        actual_keys = [call_args[0][0] for call_args in mock_txn.put.call_args_list]
        assert sorted(actual_keys) == sorted(expected_keys)

class TestPrepareEtl9gDataset:
    """Tests for the prepare_etl9g_dataset function."""
    
    @patch('src.prepare.glob')
    @patch('src.prepare.setup_lmdb')
    @patch('src.prepare.process_etl9g_files')
    @patch('src.prepare.save_metadata')
    def test_prepare_etl9g_dataset(self, mock_save, mock_process, mock_setup, mock_glob, temp_output_dir):
        """Test preparing the ETL9G dataset."""
        # Set up mocks
        mock_glob.return_value = ['file1.bin', 'file2.bin']
        
        mock_env = MagicMock()
        mock_index = defaultdict(int)
        mock_setup.return_value = (mock_env, mock_index)
        
        mock_process.return_value = 100  # 100 records processed
        
        # Call the function
        env, total = prepare_etl9g_dataset(temp_output_dir)
        
        # Check the results
        assert env == mock_env
        assert total == 100
        
        # Check that the mocks were called correctly
        mock_glob.assert_called_once_with('/data/ETL9G/ETL9G_*')
        mock_setup.assert_called_once()
        mock_process.assert_called_once_with(['file1.bin', 'file2.bin'], mock_env, mock_index, None)
        mock_save.assert_called_once_with(mock_env, 100, mock_index)
    
    @patch('src.prepare.glob')
    def test_prepare_etl9g_dataset_no_files(self, mock_glob, temp_output_dir):
        """Test handling when no ETL9G files are found."""
        # Set up mock to return empty list
        mock_glob.return_value = []
        
        # Call the function
        env, total = prepare_etl9g_dataset(temp_output_dir)
        
        # Check the results
        assert env is None
        assert total == 0
        
        # Check that the mock was called correctly
        mock_glob.assert_called_once_with('/data/ETL9G/ETL9G_*')
