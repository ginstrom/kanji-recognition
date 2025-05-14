"""
Unit tests for the clean.py module.
"""
import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from src.clean import (
    crop_and_pad,
    convert_to_bw,
    process_kanji_dict,
    prepare_dataset
)

class TestCropAndPad:
    """Tests for the crop_and_pad function."""
    
    def test_crop_and_pad_square_image(self, mock_image_array):
        """Test cropping and padding a square image."""
        result = crop_and_pad(mock_image_array)
        
        # Check dimensions
        assert result.shape == mock_image_array.shape
        
        # The central portion should be preserved
        # For a 128x128 image, we crop to 70% width (89.6 pixels)
        # So the central ~90 columns should be preserved
        center_x = mock_image_array.shape[1] // 2
        crop_width = int(mock_image_array.shape[1] * 0.7)
        left_margin = (mock_image_array.shape[1] - crop_width) // 2
        
        # Check that the central portion is preserved
        # Compare a sample point in the center that should be preserved
        assert result[64, center_x] == mock_image_array[64, center_x]
        
        # Check that the sides are zeroed out (black)
        assert result[64, 0] == 0
        assert result[64, 127] == 0
    
    def test_crop_and_pad_empty_image(self):
        """Test cropping and padding an empty (all zeros) image."""
        empty_img = np.zeros((128, 128), dtype=np.uint8)
        result = crop_and_pad(empty_img)
        
        # Result should also be all zeros
        assert np.all(result == 0)
        assert result.shape == empty_img.shape

class TestConvertToBW:
    """Tests for the convert_to_bw function."""
    
    def test_convert_to_bw_threshold(self, mock_image_array):
        """Test black and white conversion with thresholding."""
        # Mock the cv2.GaussianBlur function to return the input unchanged
        with patch('src.clean.cv2.GaussianBlur', return_value=mock_image_array):
            result = convert_to_bw(mock_image_array)
            
            # Check that the result is binary (only 0 and 255 values)
            unique_values = np.unique(result)
            assert len(unique_values) <= 2
            assert all(val in [0, 255] for val in unique_values)
            
            # Check that the white square in the center is now black (inverted)
            # The center of the mock image has value 200, which is > 20 threshold
            # So it should be 255 after thresholding, then 0 after inversion
            assert result[64, 64] == 0
            
            # Check that the black background is now white (inverted)
            # The corners of the mock image have value 0, which is < 20 threshold
            # So they should be 0 after thresholding, then 255 after inversion
            assert result[0, 0] == 255
    
    def test_convert_to_bw_empty_image(self):
        """Test black and white conversion of an empty image."""
        empty_img = np.zeros((128, 128), dtype=np.uint8)
        
        # Mock the cv2.GaussianBlur function to return the input unchanged
        with patch('src.clean.cv2.GaussianBlur', return_value=empty_img):
            result = convert_to_bw(empty_img)
            
            # An all-black input should become all-white after inversion
            assert np.all(result == 255)

class TestProcessKanjiDict:
    """Tests for the process_kanji_dict function."""
    
    def test_process_kanji_dict(self, mock_kanji_dict):
        """Test processing a kanji dictionary."""
        # Mock the convert_to_bw function
        with patch('src.clean.convert_to_bw', return_value=np.ones((128, 128), dtype=np.uint8) * 255):
            result = process_kanji_dict(mock_kanji_dict)
            
            # Check that the original dictionary is not modified
            assert id(result) != id(mock_kanji_dict)
            
            # Check that the original fields are preserved
            assert 'original' in result
            assert 'cropped' in result
            assert 'character' in result
            assert result['character'] == mock_kanji_dict['character']
            
            # Check that the twoBit field is added
            assert 'twoBit' in result
            assert isinstance(result['twoBit'], Image.Image)
            
            # Check that the twoBit image has the expected dimensions
            assert result['twoBit'].size == (128, 128)

class TestPrepareDataset:
    """Tests for the prepare_dataset function."""
    
    def test_prepare_dataset(self):
        """Test preparing a dataset of kanji dictionaries."""
        # Create mock kanji dictionaries
        mock_dicts = [
            {
                'original': Image.fromarray(np.zeros((128, 128), dtype=np.uint8)),
                'cropped': Image.fromarray(np.zeros((128, 128), dtype=np.uint8)),
                'character': '漢'
            },
            {
                'original': Image.fromarray(np.zeros((128, 128), dtype=np.uint8)),
                'cropped': Image.fromarray(np.zeros((128, 128), dtype=np.uint8)),
                'character': '字'
            }
        ]
        
        # Mock the process_kanji_dict function
        with patch('src.clean.process_kanji_dict') as mock_process:
            # Set up the mock to add a 'twoBit' field
            mock_process.side_effect = lambda d: {**d, 'twoBit': Image.fromarray(np.zeros((128, 128), dtype=np.uint8))}
            
            # Mock the ProcessPoolExecutor
            with patch('src.clean.ProcessPoolExecutor') as mock_executor:
                # Set up the executor to use our mocked process_kanji_dict
                mock_executor.return_value.__enter__.return_value.map.return_value = [
                    {**d, 'twoBit': Image.fromarray(np.zeros((128, 128), dtype=np.uint8))}
                    for d in mock_dicts
                ]
                
                # Call the function
                result = prepare_dataset(mock_dicts)
                
                # Check that we got the expected number of results
                assert len(result) == len(mock_dicts)
                
                # Check that each result has a 'twoBit' field
                assert all('twoBit' in d for d in result)
                
                # Check that the executor was called with the right arguments
                mock_executor.assert_called_once()
                mock_executor.return_value.__enter__.return_value.map.assert_called_once()
