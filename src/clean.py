import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import cv2

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

def convert_to_bw(img_array):
    """
    Convert a grayscale image to binary using a simple threshold with
    slight smoothing to reduce jagged edges, then invert.
    
    Args:
        img_array: Numpy array of the grayscale image
    
    Returns:
        binary_array: Numpy array of the black and white image
    """
    # Apply a very slight blur to reduce jagged edges
    # Using a small sigma value to maintain detail while softening edges
    smoothed = cv2.GaussianBlur(img_array, (3, 3), 0.5)
    
    # Apply a simple threshold - more sensitive than before
    # This is a bit more forgiving than the strict > 0 threshold
    binary = np.where(smoothed > 20, 255, 0).astype(np.uint8)
    
    # Invert (flip black and white)
    binary = 255 - binary
    
    return binary

def process_kanji_dict(kanji_dict):
    """
    Process a single kanji dictionary, adding a 'twoBit' field with the black and white image.
    
    Args:
        kanji_dict: Dictionary with 'original', 'cropped', and 'character' fields
        
    Returns:
        Updated dictionary with 'twoBit' field
    """
    # Make a copy of the dictionary to avoid modifying the original
    result = kanji_dict.copy()
    
    # Convert the cropped image to numpy array
    cropped_array = np.array(kanji_dict['cropped'])
    
    # Apply the simple black and white conversion
    bw_array = convert_to_bw(cropped_array)
    
    # Convert back to PIL Image
    result['twoBit'] = Image.fromarray(bw_array)
    
    return result

def prepare_dataset(kanji_dicts, max_workers=None):
    """
    Process a list of kanji dictionaries in parallel, adding 'twoBit' field to each.
    
    Args:
        kanji_dicts: List of dictionaries, each with 'original', 'cropped', and 'character' fields
        max_workers: Maximum number of worker processes (None = CPU count)
    
    Returns:
        List of updated dictionaries, each with 'twoBit' field added
    """
    print(f"Preparing {len(kanji_dicts)} kanji images")
    
    # Process dictionaries in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show progress bar
        results = list(tqdm(
            executor.map(process_kanji_dict, kanji_dicts),
            total=len(kanji_dicts),
            desc="Converting cropped images to black and white"
        ))
    
    print(f"Processed {len(results)} kanji images")
    return results

if __name__ == "__main__":
    # This is a demonstration of how to use the functions
    # when integrated with the parse_sample.py script
    
    print("This script is designed to be imported and used with parse_sample.py")
    print("Example usage:")
    print("  from parse_sample import extract_etl9g_images")
    print("  from prepare import prepare_dataset")
    print("  parsed_data = extract_etl9g_images('/data/ETL9G/ETL9G_01', limit=100)")
    print("  processed_data = prepare_dataset(parsed_data)")
