"""
Font-based Kanji Image Generator
This module generates 128x128 pixel images of Japanese characters using
available Japanese-capable fonts on the system, in regular, bold, and italic styles.

The module loads characters from an existing LMDB database (created by prepare.py)
and generates font-based images only for those characters. This ensures consistency
between the handwritten and font-based datasets.
"""
import os
import json
import lmdb
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from collections import defaultdict
from tqdm import tqdm
import unicodedata
import logging
from typing import Dict, List, Tuple, Optional, Any

# Import functions from existing modules
from clean import convert_to_bw

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_japanese_fonts():
    """
    Find all Japanese-capable fonts on the system using fc-list.
    Returns a dict mapping family -> {'regular': path, 'bold': path, 'italic': path}
    """
    import subprocess
    fonts = {}
    try:
        logger.info("Running fc-list to find Japanese fonts...")
        result = subprocess.run(
            ["fc-list", ":lang=ja"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0 or not result.stdout.strip():
            logger.error("fc-list failed or returned no output")
            return {}

        for line in result.stdout.splitlines():
            parts = [p.strip() for p in line.split(":")]
            if len(parts) < 2:
                continue

            font_path = parts[0]
            # first colon-field is the font file, second is comma-sep families
            families = parts[1].split(",")
            family = families[0]

            # find style=… in any of the remaining fields
            style = "regular"
            for seg in parts[2:]:
                seg_low = seg.lower()
                if seg_low.startswith("style="):
                    style = seg_low.split("=", 1)[1]
                    break

            style = style if style in ("regular", "bold", "italic", "oblique") else "regular"
            if style == "oblique":
                style = "italic"

            if family not in fonts:
                fonts[family] = {"regular": None, "bold": None, "italic": None}
            fonts[family][style] = font_path

        # keep only families that have a regular face
        filtered = {fam: s for fam, s in fonts.items() if s["regular"]}
        logger.info(f"Found {len(filtered)} Japanese font families with Regular variants")
        for fam, s in list(filtered.items())[:3]:
            logger.info(f"  • {fam}: {s}")
        return filtered

    except Exception:
        logger.exception("Error finding Japanese fonts")
        return {}

def get_characters_from_lmdb(db_path):
    """
    Get the set of unique characters from an existing LMDB database.
    
    Args:
        db_path: Path to the LMDB database
        
    Returns:
        list: List of unique characters found in the database
    """
    if not os.path.exists(db_path):
        logger.error(f"LMDB database not found at {db_path}")
        return []
    
    env = None
    try:
        # Open the LMDB environment
        env = lmdb.open(
            db_path, 
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        
        # Read metadata
        with env.begin() as txn:
            metadata_bytes = txn.get(b'__metadata__')
            if metadata_bytes is None:
                logger.error(f"No metadata found in LMDB database at {db_path}")
                return []
            
            metadata = json.loads(metadata_bytes.decode())
            
            # Look for character counts in metadata
            # Different databases might use different keys for character counts
            char_counts = None
            for key in ['character_counts', 'character_counts_etl9g']:
                if key in metadata:
                    char_counts = metadata[key]
                    break
            
            if not char_counts:
                logger.error(f"No character counts found in metadata for {db_path}")
                return []
            
            # Get unique characters
            characters = list(char_counts.keys())
            logger.info(f"Found {len(characters)} unique characters in {db_path}")
            
            # Make a copy of the characters
            characters_copy = characters.copy()
            
            return characters_copy
            
    except lmdb.Error as e:
        logger.error(f"Error opening LMDB database at {db_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error reading characters from {db_path}: {e}")
        return []
    finally:
        # Ensure the environment is closed even if an exception occurs
        if env is not None:
            env.close()

def render_character(char, font_path, size=(128, 128), font_size=80):
    """
    Render a character using the specified font.
    
    Args:
        char: The character to render
        font_path: Path to the font file
        size: Size of the output image (width, height)
        font_size: Base font size to use
        
    Returns:
        PIL.Image: The rendered character image
    """
    # Create a new white image
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    
    # Load the font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        logger.error(f"Error loading font {font_path}: {e}")
        return img
    
    # Get text size and position it in the center
    try:
        # Get the bounding box of the text
        try:
            # For newer PIL versions
            left, top, right, bottom = font.getbbox(char)
        except AttributeError:
            # For older PIL versions
            left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
            
        text_width = right - left
        text_height = bottom - top
        
        # Adjust position to center the text
        position = ((size[0] - text_width) // 2 - left, 
                   (size[1] - text_height) // 2 - top)
        
        # Draw the character
        draw.text(position, char, font=font, fill=0)
    except Exception as e:
        logger.error(f"Error rendering character {char} with font {font_path}: {e}")
    
    return img

def process_character_image(img, character):
    """
    Process a character image to match the format expected by the model.
    
    Args:
        img: PIL.Image of the character
        character: The character that was rendered
        
    Returns:
        dict: Dictionary with 'character' and 'twoBit' fields
    """
    # Convert PIL image to numpy array
    img_array = np.array(img)
    
    # Invert the image (white background to black, black text to white)
    img_array = 255 - img_array
    
    # Apply the same black and white conversion as in the existing pipeline
    bw_array = convert_to_bw(img_array)
    
    # Convert back to PIL Image
    bw_img = Image.fromarray(bw_array)
    
    # Create a dictionary similar to what prepare.py expects
    result = {
        'character': character,
        'twoBit': bw_img
    }
    
    return result

def setup_lmdb(output_dir, map_size=20e9):
    """
    Set up the LMDB environment and return it along with an empty index.
    
    Args:
        output_dir: Directory to store the LMDB database
        map_size: Maximum size of the database in bytes (default: 20GB)
        
    Returns:
        tuple: (LMDB environment, character index defaultdict)
    """
    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, 'kanji_fonts.lmdb')
    env = lmdb.open(db_path, map_size=int(map_size))
    index = defaultdict(int)
    return env, index

def process_and_store_kanji(kanji_dict, txn, index, font_family, style):
    """
    Process a kanji dictionary and store it in the LMDB database.
    
    Args:
        kanji_dict: Dictionary with character and image data
        txn: LMDB transaction object
        index: Character index defaultdict
        font_family: Name of the font family
        style: Font style ('regular', 'bold', or 'italic')
        
    Returns:
        str: The key used to store the data
    """
    character = kanji_dict['character']
    
    # Get the black and white image
    bw_img = kanji_dict['twoBit']
    
    # Convert PIL image to bytes for storage
    img_bytes = bw_img.tobytes()
    
    # Create a unique key using the character, its index, font family, and style
    key = f"{character}_{index[character]}_{font_family}_{style}"
    
    # Store the image bytes directly in LMDB
    txn.put(key.encode(), img_bytes)
    
    # Update the index
    index[character] += 1
    
    return key

def generate_font_kanji_dataset(output_dir, characters, limit_fonts=None):
    """
    Generate a dataset of kanji characters rendered with system fonts.
    
    Args:
        output_dir: Directory to store the LMDB database
        characters: List of characters to process from the LMDB database
        limit_fonts: Optional limit on the number of font families to process
        
    Returns:
        tuple: (LMDB environment, total records processed)
    """
    import time
    from datetime import timedelta
    
    # Get Japanese fonts
    logger.info("Finding Japanese-capable fonts...")
    japanese_fonts = get_japanese_fonts()
    logger.info(f"Found {len(japanese_fonts)} Japanese font families")
    
    if limit_fonts:
        # Take a subset of fonts
        font_families = list(japanese_fonts.keys())[:limit_fonts]
        japanese_fonts = {k: japanese_fonts[k] for k in font_families}
    
    # Use the provided character list
    all_chars = characters
    logger.info(f"Using {len(all_chars):,} characters from provided list")
    
    # Set up LMDB
    env, index = setup_lmdb(output_dir)
    
    # Calculate total expected items
    total_styles = sum(1 for styles in japanese_fonts.values() 
                      for style_name in styles if styles[style_name] is not None)
    total_expected = len(all_chars) * total_styles
    logger.info(f"Expected total items: {total_expected:,}")
    
    # Process each font family
    total_processed = 0
    font_metadata = {}
    start_time = time.time()
    last_update_time = start_time
    update_interval = 60  # Update progress every 60 seconds
    
    try:
        # Use tqdm for overall progress tracking
        with tqdm(total=total_expected, desc="Overall progress") as pbar:
            for family, styles in japanese_fonts.items():
                font_metadata[family] = {'styles': {}}
                
                for style_name, font_path in styles.items():
                    if font_path is None:
                        continue
                    
                    font_metadata[family]['styles'][style_name] = font_path
                    logger.info(f"Processing {family} {style_name} with {len(all_chars):,} characters")
                    
                    # Process each character with this font and style
                    for char in tqdm(all_chars, desc=f"Rendering {family} {style_name}", leave=False):
                        # Render the character
                        img = render_character(char, font_path)
                        
                        # Process the image
                        kanji_dict = process_character_image(img, char)
                        
                        # Store in LMDB
                        with env.begin(write=True) as txn:
                            try:
                                key = process_and_store_kanji(kanji_dict, txn, index, family, style_name)
                                total_processed += 1
                                pbar.update(1)
                                
                                # Print progress and estimated time remaining periodically
                                current_time = time.time()
                                if current_time - last_update_time > update_interval:
                                    elapsed = current_time - start_time
                                    items_per_second = total_processed / elapsed if elapsed > 0 else 0
                                    remaining_items = total_expected - total_processed
                                    estimated_remaining_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
                                    
                                    logger.info(f"Processed {total_processed:,}/{total_expected:,} items ({total_processed/total_expected*100:.1f}%)")
                                    logger.info(f"Processing rate: {items_per_second:.2f} items/second")
                                    logger.info(f"Elapsed time: {timedelta(seconds=int(elapsed))}")
                                    logger.info(f"Estimated time remaining: {timedelta(seconds=int(estimated_remaining_seconds))}")
                                    
                                    last_update_time = current_time
                                    
                            except Exception as e:
                                logger.error(f"Error storing character {char} with font {family} {style_name}: {e}")
    
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user. Saving progress...")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.info("Saving progress up to this point...")
    
    # Save metadata
    try:
        with env.begin(write=True) as txn:
            metadata = {
                'total_records': total_processed,
                'unique_characters': len(index),
                'font_families': font_metadata,
                'character_counts': {k: v for k, v in index.items()},
                'sources': ['font']
            }
            txn.put(b'__metadata__', json.dumps(metadata).encode())
            logger.info("Metadata successfully saved to LMDB")
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
        # Write to a separate file as backup
        metadata_path = os.path.join(os.path.dirname(env.path()), 'font_metadata.json')
        with open(metadata_path, 'w') as f:
            metadata = {
                'total_records': total_processed,
                'unique_characters': len(index),
                'font_families': font_metadata,
                'character_counts': {k: v for k, v in index.items()},
                'sources': ['font']
            }
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Metadata saved to {metadata_path} as backup")
    
    logger.info(f"Total records processed: {total_processed}")
    logger.info(f"Total unique characters: {len(index)}")
    
    return env, total_processed


def main(test_mode=False):
    """
    Main entry point for the script.
    
    Args:
        test_mode: If True, don't exit on errors (for testing purposes)
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate font-based kanji images')
    parser.add_argument('--output-dir', type=str, default='/app/output/prep',
                        help='Directory to store output LMDB database')
    parser.add_argument('--limit-fonts', type=int, default=None,
                        help='Limit the number of font families to process (default: use all fonts)')
    parser.add_argument('--source-lmdb', type=str, default='/app/output/prep/kanji.lmdb',
                        help='Path to source LMDB database to get characters from (default: /app/output/prep/kanji.lmdb)')
    args = parser.parse_args()
    
    # Set up the output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting font-based kanji image generation")
    
    if args.limit_fonts:
        logger.info(f"Limiting to {args.limit_fonts} font families")
    else:
        logger.info("Processing all available font families")
    
    # Get Japanese fonts
    japanese_fonts = get_japanese_fonts()
    
    # Check if Japanese fonts were found
    if not japanese_fonts:
        logger.error("No Japanese fonts found. Cannot proceed with dataset generation.")
        logger.error("Please install Japanese fonts (e.g., fonts-noto-cjk) and try again.")
        logger.error("You can check available Japanese fonts with: fc-list :lang=ja")
        if not test_mode:
            import sys
            sys.exit(1)  # Exit with error code
        return  # Return early in test mode
    
    logger.info(f"Found {len(japanese_fonts)} Japanese font families. Generating dataset...")
    
    # Get characters from the source LMDB database
    logger.info(f"Reading characters from source LMDB database: {args.source_lmdb}")
    characters = get_characters_from_lmdb(args.source_lmdb)
    if not characters:
        logger.error("Failed to get characters from source LMDB database. Exiting.")
        return
    
    logger.info(f"Using {len(characters)} unique characters from source LMDB")
    
    # Generate the dataset
    env, total_processed = generate_font_kanji_dataset(
        output_dir, 
        characters=characters,
        limit_fonts=args.limit_fonts
    )
    
    if env is None:
        logger.error("Failed to generate dataset. Exiting.")
        return
    
    # Close the LMDB environment
    env.close()
    
    logger.info("\nDataset generation complete!")
    logger.info(f"Processed {total_processed} records")
    logger.info(f"LMDB database saved to: {os.path.join(output_dir, 'kanji_fonts.lmdb')}")


if __name__ == "__main__":
    main()
