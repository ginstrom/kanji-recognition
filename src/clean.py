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
    Convert to binary using multiple approaches and select the best one.
    
    Args:
        img_array: Numpy array of the grayscale image
        
    Returns:
        binary_array: Numpy array of the black and white image
    """
    # Ensure proper data type and range
    img_array = img_array.astype(np.uint8)
    
    # Apply slight blur for noise reduction
    smoothed = cv2.GaussianBlur(img_array, (3, 3), 0.5)
    
    # Method 1: Otsu's thresholding
    _, binary_otsu = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Method 2: Adaptive thresholding
    binary_adaptive = cv2.adaptiveThreshold(
        smoothed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    
    # Method 3: Fixed value thresholding (your original approach)
    binary_fixed = np.where(smoothed > 20, 255, 0).astype(np.uint8)
    
    # Choose the method that preserves the most stroke information
    # Here we select the method that results in an appropriate amount of foreground pixels
    # This is based on the assumption that kanji characters typically occupy 10-40% of image area
    
    def calculate_pixel_ratio(img):
        """Calculate ratio of foreground (white) pixels"""
        return np.sum(img == 255) / img.size
    
    # Invert all binary images so strokes are white (255)
    binary_otsu = 255 - binary_otsu
    binary_adaptive = 255 - binary_adaptive
    binary_fixed = 255 - binary_fixed
    
    # Calculate white pixel ratios
    ratio_otsu = calculate_pixel_ratio(binary_otsu)
    ratio_adaptive = calculate_pixel_ratio(binary_adaptive)
    ratio_fixed = calculate_pixel_ratio(binary_fixed)
    
    # Choose method based on reasonable pixel density for a kanji character
    # These thresholds can be adjusted based on your specific dataset
    if 0.1 <= ratio_otsu <= 0.4:
        return binary_otsu
    elif 0.1 <= ratio_adaptive <= 0.4:
        return binary_adaptive
    else:
        # Apply morphological operations to improve the fixed threshold result
        kernel = np.ones((2, 2), np.uint8)
        binary_fixed = cv2.morphologyEx(binary_fixed, cv2.MORPH_CLOSE, kernel)
        return binary_fixed

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
