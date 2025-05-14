"""
Shared pytest fixtures for the kanji recognition project tests.
"""
import os
import io
import tempfile
import shutil
import struct
import pytest
import numpy as np
from PIL import Image
import lmdb

@pytest.fixture
def mock_binary_stream():
    """
    Create a mock binary stream with fixed-size records for testing.
    """
    # Create a binary stream with 3 records of 100 bytes each
    data = b''.join([f"RECORD{i:03d}".encode().ljust(100, b'\x00') for i in range(3)])
    return io.BytesIO(data)

@pytest.fixture
def mock_etl9g_record():
    """
    Create a mock ETL9G record with known metadata and image data.
    """
    # Create header with JIS code 0x3441 (corresponds to a known kanji)
    header = struct.pack('>HH6xH6x4s4x', 1, 0x3441, 0, b'TEST')
    
    # Create a simple 128x128 test image (checkerboard pattern)
    img_data = np.zeros((64, 64), dtype=np.uint8)
    img_data[::2, ::2] = 255  # Set alternate pixels to white
    img_data[1::2, 1::2] = 255
    
    # Convert to the format expected by ETL9G (pairs of 4-bit values)
    flat_data = img_data.flatten()
    packed_data = bytearray()
    for i in range(0, len(flat_data), 2):
        if i+1 < len(flat_data):
            hi = flat_data[i] // 17
            lo = flat_data[i+1] // 17
            packed_data.append((hi << 4) | lo)
        else:
            hi = flat_data[i] // 17
            packed_data.append(hi << 4)
    
    # Pad to 8128 bytes
    packed_data = packed_data.ljust(8128, b'\x00')
    
    # Combine header and image data
    record = header + bytes(packed_data)
    
    # Pad to ETL9G_RECORD_SIZE if needed
    if len(record) < 8199:
        record = record.ljust(8199, b'\x00')
    
    return record

@pytest.fixture
def mock_image_array():
    """
    Create a mock 128x128 numpy array representing a grayscale image.
    """
    # Create a simple test pattern (a centered square)
    img = np.zeros((128, 128), dtype=np.uint8)
    img[32:96, 32:96] = 200  # Create a white square in the center
    return img

@pytest.fixture
def mock_kanji_dict():
    """
    Create a mock kanji dictionary with original and cropped images.
    """
    # Create a simple test image
    img = np.zeros((128, 128), dtype=np.uint8)
    img[32:96, 32:96] = 200  # Create a white square in the center
    
    original_img = Image.fromarray(img, mode='L')
    cropped_img = Image.fromarray(img, mode='L')  # Same image for simplicity
    
    return {
        'original': original_img,
        'cropped': cropped_img,
        'character': '漢'  # Example kanji character
    }

@pytest.fixture
def temp_output_dir():
    """
    Create a temporary directory for test outputs.
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_lmdb_env(temp_output_dir):
    """
    Create a temporary LMDB environment for testing.
    """
    db_path = os.path.join(temp_output_dir, 'test.lmdb')
    env = lmdb.open(db_path, map_size=10485760)  # 10MB
    yield env
    # Cleanup
    env.close()
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
