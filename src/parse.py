import struct
import numpy as np
from PIL import Image
import os
from pathlib import Path
from glob import glob
from jis_unicode_map import jis_to_unicode

# import the local clean module
from clean import crop_and_pad, process_kanji_dict

ETL9G_RECORD_SIZE = 8199

def read_records(input_stream, record_size):
    """
    Generator function that reads records of a specified size from a binary input stream.
    
    Args:
        input_stream: A binary file-like object opened in 'rb' mode
        record_size: Size of each record in bytes
        
    Yields:
        bytes: Each record blob until the stream is exhausted
    """
    while True:
        record = input_stream.read(record_size)
        
        # Check if we've reached EOF (empty string) or got an incomplete record
        if not record:
            break
            
        if len(record) != record_size:
            print(f"Warning: Incomplete record read. Expected {record_size} bytes, got {len(record)} bytes.")
            break
            
        yield record

# Save output to a folder for viewing in browser
# This path should match the volume mount in docker-compose.yml for output
output_dir_base = '/app/output/prep/etl9g_images' 

def jis2unicode(jis_code):
    return jis_to_unicode.get(jis_code, 'UNK')

def extract_item9g_image(s):
    """
    Extracts image data and character from a single ETL9G record.
    Returns a dictionary with 'original' image, 'cropped' image, and 'character'.
    """
    # Read metadata
    r = struct.unpack('>HH6xH6x4s4x', s[:26])
    # print('r: ', r) # Keep or remove print based on debugging needs
    jis_code = r[1]

    # Extract the last 8128 bytes as image data
    img_data = s[-8128:]
    if len(img_data) != 8128:
        # Consider raising an error or returning None/empty dict
        print(f"Invalid image data length: expected 8128 bytes, got {len(img_data)}")
        return None 

    raw = np.frombuffer(img_data, dtype=np.uint8)
    # Create a 128x128 array (square) instead of 127x128
    pixels = np.zeros((128, 128), dtype=np.uint8)

    # Fill the first 127 rows with the actual data
    for j in range(8128):
        byte = raw[j]
        hi = (byte >> 4) * 17
        lo = (byte & 0x0F) * 17
        y = (j * 2) // 128
        x = (j * 2) % 128
        # Ensure we don't go out of bounds
        if y < 128 and x < 128:
            pixels[y, x] = hi
        if y < 128 and x + 1 < 128:
            pixels[y, x + 1] = lo
    
    original_img = Image.fromarray(pixels, mode='L')
    
    # Apply crop and pad technique
    processed_pixels = crop_and_pad(pixels)
    cropped_img = Image.fromarray(processed_pixels, mode='L')
    
    unicode_char = jis2unicode(jis_code)
    
    return {
        'original': original_img,
        'cropped': cropped_img,
        'character': unicode_char
    }

def extract_etl9g_images(file_paths, limit=20):
    """
    Generator function that extracts images from ETL9G dataset files and yields each dictionary as it's created.
    No longer saves images to disk.
    
    Args:
        file_paths: List of paths to ETL9G data files
        limit: Optional limit on the number of records to process per file
        
    Yields:
        dict: Dictionary with 'original', 'cropped', 'twoBit', and 'character' fields
    """
    for num, file_path in enumerate(file_paths):
        print(f"Processing file {num + 1}/{len(file_paths)}: {file_path}")
        # Read the binary file
        with open(file_path, 'rb') as f:
            record_idx = 0
            for s in read_records(f, ETL9G_RECORD_SIZE):
                if limit is not None and record_idx >= limit:
                    break

                try:
                    item_data = extract_item9g_image(s)

                    if item_data is None:
                        print(f"Skipping record {record_idx} due to image data error.")
                        record_idx += 1
                        continue

                    item_data = process_kanji_dict(item_data)
                    yield item_data
                    record_idx += 1
                except Exception as e:
                    print(f"Error processing record {record_idx}: {e}")
                    record_idx += 1
                    continue

def clean_output_dir(base_dir):
    """
    This function is kept for backward compatibility but no longer needed
    since we're not writing files to disk anymore.
    """
    print("No longer writing images to disk, skipping directory cleaning.")

def main():
    # Example usage (adjust path and limit as needed):
    image_files = glob(f'/data/ETL9G/ETL9G_*')
    if not image_files:
        print("No ETL9G files found. Please check the path '/data/ETL9G/'.")
        return
    print(f"Found {len(image_files)} ETL9G files.")
    print("Extracting images from ETL9G dataset...")
    
    # Example of using the generator
    count = 0
    for item in extract_etl9g_images(image_files, limit=10):
        count += 1
        # Process each item as it's yielded
        # For example, you could print the character:
        print(f"Processed item {count}: character {item['character']}")
    
    print(f"Extracted {count} items without writing to disk.")
    print("To store these items in LMDB, use prepare.py")

if __name__ == "__main__":
    main()
