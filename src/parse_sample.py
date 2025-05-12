import struct
import numpy as np
from PIL import Image
import os
from pathlib import Path
from glob import glob
from jis_unicode_map import jis_to_unicode

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
output_dir_base = '/app/etl9g_images' 

def jis2unicode(jis_code):
    return jis_to_unicode.get(jis_code, 'UNK')

def crop_and_pad(img_array):
    """
    Crop the central portion of the image and add padding to maintain aspect ratio.
    """
    height, width = img_array.shape
    
    # Take the central 70% of the image width
    crop_width = int(width * 0.7)
    left_margin = (width - crop_width) // 2
    
    # Crop the central portion
    cropped = img_array[:, left_margin:left_margin+crop_width]
    
    # Create a new black image with the original dimensions
    padded = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate padding
    pad_left = (width - crop_width) // 2
    
    # Place the cropped image in the center of the padded image
    padded[:, pad_left:pad_left+crop_width] = cropped
    
    return padded

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
    Extracts images from ETL9G dataset files and saves them to the output directory.
    """
    for num, file_path in enumerate(file_paths):
        print(f"Processing file {num + 1}/{len(file_paths)}: {file_path}")
        output_dir = os.path.join(output_dir_base, str(num))
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        # Read the binary file
        with open(file_path, 'rb') as f:
            record_idx = 0
            for s in read_records(f, ETL9G_RECORD_SIZE):
                if record_idx >= limit:
                    break

                try:
                    item_data = extract_item9g_image(s)

                    if item_data is None:
                        print(f"Skipping record {record_idx} due to image data error.")
                        continue

                    original_img = item_data['original']
                    cropped_img = item_data['cropped']
                    unicode_char = item_data['character']
                    
                    # Save original image for comparison
                    original_path = f"{output_dir}/{record_idx:05}_original.png"
                    original_img.save(original_path)
                    
                    # Save cropped image
                    output_path = f"{output_dir}/{record_idx:05}_{unicode_char}.png"
                    # print("output_path: ", output_path) # Keep or remove print
                    cropped_img.save(output_path)

                    record_idx += 1
                except Exception as e:
                    print(f"Error processing record {record_idx}: {e}")
                    # It might be better to increment record_idx here or handle EOF differently
                    # to avoid an infinite loop if the last record is problematic.
                    # For now, continue will skip to the next read attempt.
                    continue

def clean_output_dir(base_dir):
    """Deletes all files within subdirectories of the base output directory."""
    print("Deleting existing images in output directory...")
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist. Skipping deletion.")
        return

    for entry in os.listdir(base_dir):
        output_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(output_dir):
            continue
        print(f"Cleaning directory: {output_dir}")
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            # Skip directories if any exist inside
            if os.path.isdir(file_path):
                continue
            try:
                os.remove(file_path)
                # print(f"Deleted file: {file_path}") # Optional: for verbose logging
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

def main():
    # Example usage (adjust path and limit as needed):
    # first delete any existing images in the output directory
    clean_output_dir(output_dir_base)
    # image_files = glob('/data/ETL9G/ETL9G_*') # Original glob pattern
    image_files = glob(f'/data/ETL9G/ETL9G_*') # Corrected glob pattern if needed, ensure path exists
    if not image_files:
        print("No ETL9G files found. Please check the path '/data/ETL9G/'.")
        return
    print(f"Found {len(image_files)} ETL9G files.")
    print("Extracting images from ETL9G dataset...")
    extract_etl9g_images(image_files, limit=10)

if __name__ == "__main__":
    main()
