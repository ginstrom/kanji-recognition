"""
Unit tests for the parse_etl9g.py module.
"""
import io
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from src.parse_etl9g import (
    read_records,
    extract_image,
    jis2unicode,
    extract_images,
    ETL9G_RECORD_SIZE
)

class TestReadRecords:
    """Tests for the read_records function."""
    
    def test_read_records_complete(self, mock_binary_stream):
        """Test reading complete records from a binary stream."""
        records = list(read_records(mock_binary_stream, 100))
        assert len(records) == 3
        assert records[0].startswith(b'RECORD000')
        assert records[1].startswith(b'RECORD001')
        assert records[2].startswith(b'RECORD002')
        assert all(len(record) == 100 for record in records)
    
    def test_read_records_incomplete(self):
        """Test handling of incomplete records."""
        # Create a stream with an incomplete final record
        data = b''.join([b'A' * 100, b'B' * 100, b'C' * 50])
        stream = io.BytesIO(data)
        
        records = list(read_records(stream, 100))
        assert len(records) == 2  # Should only read the complete records
        assert records[0] == b'A' * 100
        assert records[1] == b'B' * 100
    
    def test_read_records_empty(self):
        """Test reading from an empty stream."""
        stream = io.BytesIO(b'')
        records = list(read_records(stream, 100))
        assert len(records) == 0

class TestJis2Unicode:
    """Tests for the jis2unicode function."""
    
    def test_known_mapping(self):
        """Test conversion of known JIS codes to Unicode."""
        # This will depend on your jis_to_unicode mapping
        # Replace with actual known mappings from your project
        with patch('src.parse_etl9g.jis_to_unicode', {0x3441: '漢'}):
            assert jis2unicode(0x3441) == '漢'
    
    def test_unknown_mapping(self):
        """Test handling of unknown JIS codes."""
        with patch('src.parse_etl9g.jis_to_unicode', {}):
            assert jis2unicode(0x9999) == 'UNK'  # Should return 'UNK' for unknown codes

class TestExtractImage:
    """Tests for the extract_image function."""
    
    def test_extract_image_valid(self, mock_etl9g_record):
        """Test extracting image data from a valid ETL9G record."""
        with patch('src.parse_etl9g.jis2unicode', return_value='漢'):
            with patch('src.parse_etl9g.crop_and_pad', return_value=np.zeros((128, 128), dtype=np.uint8)):
                result = extract_image(mock_etl9g_record)
                
                assert result is not None
                assert 'original' in result
                assert 'cropped' in result
                assert 'character' in result
                assert result['character'] == '漢'
                assert isinstance(result['original'], Image.Image)
                assert isinstance(result['cropped'], Image.Image)
                assert result['original'].size == (128, 128)
                assert result['cropped'].size == (128, 128)
    
    def test_extract_image_invalid_data(self):
        """Test handling of invalid image data."""
        # Create a record that's too short
        invalid_record = b'x' * 100  # Much shorter than required
        
        result = extract_image(invalid_record)
        assert result is None

class TestExtractImages:
    """Tests for the extract_images function."""
    
    @patch('src.parse_etl9g.open')
    @patch('src.parse_etl9g.read_records')
    @patch('src.parse_etl9g.extract_image')
    @patch('src.parse_etl9g.process_kanji_dict')
    def test_extract_images_success(self, mock_process, mock_extract, mock_read, mock_open):
        """Test successful extraction of images from ETL9G files."""
        # Setup mocks
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock read_records to return 2 records
        mock_read.return_value = [b'record1', b'record2']
        
        # Mock extract_image to return a valid dictionary
        mock_extract.return_value = {'original': MagicMock(), 'cropped': MagicMock(), 'character': '漢'}
        
        # Mock process_kanji_dict to return the input with an added 'twoBit' field
        mock_process.side_effect = lambda x: {**x, 'twoBit': MagicMock()}
        
        # Call the function with a list of file paths
        results = list(extract_images(['file1.bin', 'file2.bin'], limit=2))
        
        # Verify results
        assert len(results) == 4  # 2 files * 2 records
        assert all('twoBit' in result for result in results)
        assert all(result['character'] == '漢' for result in results)
        
        # Verify mock calls
        assert mock_open.call_count == 2
        assert mock_read.call_count == 2
        assert mock_extract.call_count == 4
        assert mock_process.call_count == 4
    
    @patch('src.parse_etl9g.open')
    @patch('src.parse_etl9g.read_records')
    @patch('src.parse_etl9g.extract_image')
    @patch('src.parse_etl9g.process_kanji_dict')
    def test_extract_images_error_handling(self, mock_process, mock_extract, mock_read, mock_open):
        """Test error handling during image extraction."""
        # Setup mocks
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Mock read_records to return 2 records
        mock_read.return_value = [b'record1', b'record2']
        
        # Mock extract_image to raise an exception for the second record
        mock_extract.side_effect = [
            {'original': MagicMock(), 'cropped': MagicMock(), 'character': '漢'},
            Exception("Test error")
        ]
        
        # Mock process_kanji_dict to return the input with an added 'twoBit' field
        mock_process.side_effect = lambda x: {**x, 'twoBit': MagicMock()}
        
        # Call the function
        results = list(extract_images(['file1.bin'], limit=2))
        
        # Verify results - should only get one result since the second raises an exception
        assert len(results) == 1
        assert results[0]['character'] == '漢'
