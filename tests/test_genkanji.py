"""
Tests for the genkanji.py module.
"""
import os
import pytest
import numpy as np
from PIL import Image
import lmdb
import json
from unittest.mock import patch, MagicMock

# Import the module to test
from src.genkanji import (
    get_japanese_fonts,
    get_characters_from_lmdb,
    render_character,
    process_character_image,
    setup_lmdb,
    process_and_store_kanji,
    generate_font_kanji_dataset,
    main
)

class TestGenKanji:
    """
    Test class for genkanji.py module.
    """
    
    @patch('subprocess.run')
    def test_get_japanese_fonts_with_fonts(self, mock_run):
        """Test get_japanese_fonts when fonts are available."""
        # Mock the subprocess.run result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc: Noto Sans CJK JP:style=Regular\n"
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc: Noto Sans CJK JP:style=Bold\n"
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc: Noto Serif CJK JP:style=Regular\n"
        )
        mock_run.return_value = mock_result
        
        # Call the function
        fonts = get_japanese_fonts()
        
        # Check the results
        assert len(fonts) == 2  # Two font families
        assert "Noto Sans CJK JP" in fonts
        assert "Noto Serif CJK JP" in fonts
        assert fonts["Noto Sans CJK JP"]["regular"] == "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        assert fonts["Noto Sans CJK JP"]["bold"] == "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        
    @patch('subprocess.run')
    def test_get_japanese_fonts_no_fonts(self, mock_run):
        """Test get_japanese_fonts when no fonts are available."""
        # Mock the subprocess.run result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        
        # Call the function
        fonts = get_japanese_fonts()
        
        # Check the results
        assert len(fonts) == 0
    
    @patch('os.path.exists')
    @patch('lmdb.open')
    @patch('json.loads')
    def test_get_characters_from_lmdb(self, mock_json_loads, mock_lmdb_open, mock_exists):
        """Test getting characters from an LMDB database."""
        # Mock os.path.exists to return True
        mock_exists.return_value = True
        
        # Mock the LMDB environment
        mock_env = MagicMock()
        mock_txn = MagicMock()
        mock_env.begin.return_value = mock_txn
        mock_lmdb_open.return_value = mock_env
        
        # Mock the metadata
        metadata = {
            'character_counts': {
                'あ': 10,
                '漢': 5
            }
        }
        mock_txn.get.return_value = b'dummy_metadata'  # Just a dummy value
        
        # Mock json.loads to return our metadata
        mock_json_loads.return_value = metadata
        
        # Call the function
        characters = get_characters_from_lmdb('/path/to/db')
        
        # Check the results
        assert len(characters) == 2
        assert 'あ' in characters
        assert '漢' in characters
        
        # Verify that the environment was closed in the finally block
        mock_env.close.assert_called_once()
        
        # Verify that we got a copy of the characters, not the original list
        assert isinstance(characters, list)
        
    def test_render_character(self):
        """Test rendering a character to an image."""
        # Create a mock font path
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        
        # Skip if the font doesn't exist
        if not os.path.exists(font_path):
            pytest.skip(f"Font {font_path} not found, skipping test")
            
        # Render a character
        img = render_character('あ', font_path)
        
        # Check the result
        assert isinstance(img, Image.Image)
        assert img.size == (128, 128)
        assert img.mode == 'L'  # Grayscale
        
    def test_process_character_image(self):
        """Test processing a character image."""
        # Create a test image
        img = Image.new('L', (128, 128), color=255)
        # Draw a simple character (a box)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([32, 32, 96, 96], fill=0)
        
        # Process the image
        result = process_character_image(img, 'あ')
        
        # Check the result
        assert 'character' in result
        assert result['character'] == 'あ'
        assert 'twoBit' in result
        assert isinstance(result['twoBit'], Image.Image)
        
    def test_setup_lmdb(self, tmpdir):
        """Test setting up an LMDB environment."""
        # Create a temporary directory
        output_dir = str(tmpdir)
        
        # Set up LMDB
        env, index = setup_lmdb(output_dir, map_size=1e6)
        
        # Check the result
        assert env is not None
        assert isinstance(index, dict)
        assert os.path.exists(os.path.join(output_dir, 'kanji_fonts.lmdb'))
        
        # Clean up
        env.close()
        
    def test_process_and_store_kanji(self, tmpdir):
        """Test processing and storing a kanji dictionary in LMDB."""
        # Create a temporary directory
        output_dir = str(tmpdir)
        
        # Set up LMDB
        env, index = setup_lmdb(output_dir, map_size=1e6)
        
        # Create a test image
        img = Image.new('L', (128, 128), color=255)
        # Draw a simple character (a box)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([32, 32, 96, 96], fill=0)
        
        # Create a kanji dictionary
        kanji_dict = {
            'character': 'あ',
            'twoBit': img
        }
        
        # Process and store the kanji
        with env.begin(write=True) as txn:
            key = process_and_store_kanji(kanji_dict, txn, index, 'test_font', 'regular')
        
        # Check the result
        assert key == 'あ_0_test_font_regular'
        assert index['あ'] == 1
        
        # Verify the data was stored
        with env.begin() as txn:
            data = txn.get(key.encode())
            assert data is not None
        
        # Clean up
        env.close()
        
    @patch('src.genkanji.get_japanese_fonts')
    def test_generate_font_kanji_dataset(self, mock_get_fonts, tmpdir):
        """Test generating a font kanji dataset."""
        # Create a temporary directory
        output_dir = str(tmpdir)
        
        # Mock the get_japanese_fonts function
        mock_get_fonts.return_value = {
            'test_font': {
                'regular': '/path/to/font.ttf',
                'bold': '/path/to/font-bold.ttf',
                'italic': None
            }
        }
        
        # Mock the render_character function to return a test image
        def mock_render(char, font_path, size=(128, 128), font_size=80):
            from PIL import ImageDraw
            img = Image.new('L', size, color=255)
            draw = ImageDraw.Draw(img)
            draw.rectangle([32, 32, 96, 96], fill=0)
            return img
            
        # Patch the render_character function
        with patch('src.genkanji.render_character', side_effect=mock_render):
            # Generate the dataset with a very small subset of characters
            characters = ['あ', '漢']
            env, total = generate_font_kanji_dataset(
                output_dir,
                characters=characters,
                limit_fonts=1
            )
        
        # Check the result
        assert env is not None
        assert total > 0
        
        # Verify the metadata was stored
        with env.begin() as txn:
            metadata_bytes = txn.get(b'__metadata__')
            assert metadata_bytes is not None
            metadata = json.loads(metadata_bytes.decode())
            assert 'total_records' in metadata
            assert metadata['total_records'] > 0
        
        # Clean up
        env.close()
        
    @patch('src.genkanji.get_japanese_fonts')
    @patch('src.genkanji.get_characters_from_lmdb')
    @patch('src.genkanji.generate_font_kanji_dataset')
    def test_main_with_fonts(self, mock_generate, mock_get_chars, mock_get_fonts):
        """Test the main function when fonts are available."""
        # Mock the get_japanese_fonts function
        mock_get_fonts.return_value = {
            'test_font': {
                'regular': '/path/to/font.ttf',
                'bold': '/path/to/font-bold.ttf',
                'italic': None
            }
        }
        
        # Mock the get_characters_from_lmdb function
        mock_get_chars.return_value = ['あ', '漢']
        
        # Mock the generate_font_kanji_dataset function
        mock_env = MagicMock()
        mock_generate.return_value = (mock_env, 10)
        
        # Call the main function in test mode
        main(test_mode=True)
        
        # Check that generate_font_kanji_dataset was called
        mock_generate.assert_called_once()
        
        # Check that env.close() was called
        mock_env.close.assert_called_once()
        
    @patch('src.genkanji.get_japanese_fonts')
    @patch('src.genkanji.generate_font_kanji_dataset')
    @patch('src.genkanji.get_characters_from_lmdb')
    def test_main_no_fonts(self, mock_get_chars, mock_generate, mock_get_fonts):
        """Test the main function when no fonts are available."""
        # Mock get_characters_from_lmdb to ensure main proceeds
        mock_get_chars.return_value = ['一', '二']

        # Mock get_japanese_fonts (called inside the real generate_font_kanji_dataset,
        # but less critical here as generate_font_kanji_dataset itself is mocked)
        mock_get_fonts.return_value = {}

        # Configure the mock for generate_font_kanji_dataset to return what the
        # real function would return if no fonts were found by its internal logic.
        mock_generate.return_value = (None, 0)
    
        # Call the main function in test mode
        # It should handle env=None gracefully and return.
        main(test_mode=True)

        # Assert that generate_font_kanji_dataset was called
        mock_generate.assert_called_once()
        # We could also check sys.argv or mock argparse if specific args were needed for main
        # For now, assume default args for main are sufficient for this test.
        # We could also capture logger output to check for the expected error message.
