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
# import matplotlib.font_manager as fm # Unused F401
from collections import defaultdict
from tqdm import tqdm
# import unicodedata # Unused F401
import logging
from typing import Dict, List, Optional, Any # Tuple unused F401
import gc # Added for garbage collection
# import traceback # Unused F401
import psutil # Added for memory monitoring
import time # Already imported, but ensure it is used for new timing/checkpoint
from datetime import timedelta # Already imported, ensure used for new timing

# Import functions from existing modules
from clean import convert_to_bw

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "genkanji_checkpoint.json" # For checkpointing

def get_memory_usage(process: psutil.Process) -> float:
    """Get current memory usage in MB and percentage."""
    mem_info = process.memory_info()
    mem_percent = process.memory_percent()
    return mem_info.rss / (1024 * 1024), mem_percent

def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint data from a JSON file."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            return checkpoint_data
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from checkpoint file {checkpoint_path}. Starting fresh or with a new checkpoint.", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}", exc_info=True)
    return None

def save_checkpoint(checkpoint_path: str, data: Dict[str, Any]):
    """Save checkpoint data to a JSON file."""
    try:
        # Create a temporary file path
        temp_checkpoint_path = checkpoint_path + ".tmp"
        with open(temp_checkpoint_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Atomically replace the old checkpoint file
        os.replace(temp_checkpoint_path, checkpoint_path)
        # logger.debug(f"Checkpoint saved to {checkpoint_path}") # Potentially too verbose for INFO
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}", exc_info=True)

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

def generate_font_kanji_dataset(output_dir: str, 
                                characters: List[str], 
                                limit_fonts: Optional[int] = None,
                                chunk_size: int = 1000,
                                resume: bool = False,
                                checkpoint_path_base: str = ".",
                                memory_warning_threshold: float = 80.0):
    """
    Generate a dataset of kanji characters rendered with system fonts.
    Includes batch processing, checkpointing, and memory monitoring.
    
    Args:
        output_dir: Directory to store the LMDB database
        characters: List of characters to process
        limit_fonts: Optional limit on the number of font families to process
        chunk_size: Number of characters to process in a single batch
        resume: Whether to resume from a checkpoint
        checkpoint_path_base: Base directory for the checkpoint file
        memory_warning_threshold: Memory usage percentage to trigger a warning
        
    Returns:
        tuple: (LMDB environment, total records processed)
    """
    process = psutil.Process(os.getpid()) # For memory monitoring
    checkpoint_file = os.path.join(checkpoint_path_base, CHECKPOINT_FILE)

    logger.info("Finding Japanese-capable fonts...")
    all_japanese_fonts = get_japanese_fonts()
    
    if not all_japanese_fonts:
        logger.error("No Japanese fonts found. Cannot generate dataset.")
        return None, 0

    font_families_to_process_dict = all_japanese_fonts
    if limit_fonts:
        limited_families = list(all_japanese_fonts.keys())[:limit_fonts]
        font_families_to_process_dict = {k: all_japanese_fonts[k] for k in limited_families}
    
    logger.info(f"Found {len(all_japanese_fonts)} Japanese font families. Processing up to {len(font_families_to_process_dict)} families.")

    all_chars = characters
    logger.info(f"Using {len(all_chars):,} characters from provided list.")
    
    env, index = setup_lmdb(output_dir) # index is a defaultdict(int)
    
    total_processed_overall = 0
    font_metadata = {} 
    
    # --- Checkpoint Loading ---
    start_font_family_name: Optional[str] = None
    start_style_name: Optional[str] = None
    processed_char_offset_in_start_style: int = 0

    if resume:
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            logger.info(f"Attempting to resume from checkpoint: {checkpoint_data}")
            start_font_family_name = checkpoint_data.get('current_font_family')
            start_style_name = checkpoint_data.get('current_style_name')
            processed_char_offset_in_start_style = checkpoint_data.get('processed_char_offset_in_current_style', 0)
            total_processed_overall = checkpoint_data.get('total_records_processed_overall', 0)
            
            # Load character counts (index)
            loaded_character_counts = checkpoint_data.get('character_counts', {})
            for char, count in loaded_character_counts.items():
                index[char] = count
            
            font_metadata = checkpoint_data.get('font_metadata_partial', {})
            logger.info(f"Resuming. Total already processed: {total_processed_overall}. Chars in index: {len(index)}")
            logger.info(f"Resume point: Font='{start_font_family_name}', Style='{start_style_name}', CharOffset={processed_char_offset_in_start_style}")
        else:
            logger.info("No valid checkpoint found or failed to load. Starting fresh.")
            resume = False # Ensure we don't try to use invalid resume points
    # --- End Checkpoint Loading ---

    # Calculate total expected items based on fonts not yet fully processed
    # This is tricky with resume. For simplicity, tqdm will use total of all possible items,
    # and `initial` will be set to `total_processed_overall`.
    total_styles_count = sum(1 for styles in font_families_to_process_dict.values() 
                             for style_path in styles.values() if style_path is not None)
    total_expected_items = len(all_chars) * total_styles_count
    logger.info(f"Total possible items to generate (across all specified fonts/styles): {total_expected_items:,}")

    start_time = time.time()
    last_update_time = start_time
    update_interval = 60 # seconds

    try:
        with tqdm(total=total_expected_items, initial=total_processed_overall, desc="Overall Progress") as pbar:
            font_families_iterable = list(font_families_to_process_dict.items())
            
            # Skip already processed font families if resuming
            if resume and start_font_family_name:
                try:
                    start_font_idx = next(i for i, (fam, _) in enumerate(font_families_iterable) if fam == start_font_family_name)
                    font_families_iterable = font_families_iterable[start_font_idx:]
                except StopIteration:
                    logger.warning(f"Resume font '{start_font_family_name}' not found in current font list. Processing all fonts.")
                    start_font_family_name = None # Process all
                    start_style_name = None
                    processed_char_offset_in_start_style = 0


            for family_name, styles in font_families_iterable:
                if family_name not in font_metadata: # Initialize if not loaded from checkpoint
                    font_metadata[family_name] = {'styles': {}}
                
                styles_iterable = list(styles.items())

                # Skip already processed styles within the starting font family
                if resume and start_font_family_name == family_name and start_style_name:
                    try:
                        start_style_idx = next(i for i, (stl, _) in enumerate(styles_iterable) if stl == start_style_name)
                        styles_iterable = styles_iterable[start_style_idx:]
                    except StopIteration:
                        logger.warning(f"Resume style '{start_style_name}' for font '{family_name}' not found. Processing all styles for this font.")
                        start_style_name = None # Process all styles from here
                        processed_char_offset_in_start_style = 0
                
                for style_name, font_path in styles_iterable:
                    if font_path is None:
                        continue
                    
                    font_metadata[family_name]['styles'][style_name] = font_path
                    
                    current_char_processing_offset = 0
                    if resume and start_font_family_name == family_name and start_style_name == style_name:
                        current_char_processing_offset = processed_char_offset_in_start_style
                        logger.info(f"Resuming font '{family_name}' style '{style_name}' from character offset {current_char_processing_offset}.")
                    else:
                        # This is a new style (or fresh start), reset char offset
                        current_char_processing_offset = 0

                    logger.info(f"Processing Font: {family_name}, Style: {style_name}. Characters: {len(all_chars)}. Starting at offset: {current_char_processing_offset}")
                    
                    # Iterate through characters in chunks for the current font/style
                    for i in range(current_char_processing_offset, len(all_chars), chunk_size):
                        char_batch = all_chars[i : i + chunk_size]
                        if not char_batch:
                            continue

                        batch_num = (i // chunk_size) + 1
                        num_batches_for_style = (len(all_chars) - current_char_processing_offset + chunk_size - 1) // chunk_size
                        
                        mem_before_mb, mem_before_percent = get_memory_usage(process)
                        logger.debug(f"Font: {family_name}, Style: {style_name}, Batch {batch_num}/{num_batches_for_style} ({len(char_batch)} chars). Mem before: {mem_before_mb:.2f}MB ({mem_before_percent:.1f}%)")

                        # LMDB Batch Write
                        num_processed_in_batch = 0
                        with env.begin(write=True) as txn:
                            for char_in_batch in char_batch:
                                try:
                                    # Render the character
                                    img = render_character(char_in_batch, font_path)
                                    # Process the image
                                    kanji_dict = process_character_image(img, char_in_batch)
                                    # Store in LMDB (process_and_store_kanji increments index[char_in_batch])
                                    process_and_store_kanji(kanji_dict, txn, index, family_name, style_name)
                                    
                                    # total_processed_overall and pbar are updated *outside* the transaction
                                    # after it commits, to reflect only successfully written items.
                                    num_processed_in_batch +=1
                                except Exception as e:
                                    logger.error(f"Error processing/storing char '{char_in_batch}' (Font: {family_name}, Style: {style_name}) in batch: {e}", exc_info=True)
                                    # Continue with other characters in the batch
                        
                        total_processed_overall += num_processed_in_batch
                        pbar.update(num_processed_in_batch)

                        mem_after_mb, mem_after_percent = get_memory_usage(process)
                        logger.debug(f"Font: {family_name}, Style: {style_name}, Batch {batch_num} processed. Mem after: {mem_after_mb:.2f}MB ({mem_after_percent:.1f}%)")
                        
                        if mem_after_percent > memory_warning_threshold:
                            logger.warning(f"Memory usage ({mem_after_percent:.1f}%) exceeds threshold ({memory_warning_threshold:.1f}%) after batch.")

                        gc.collect() # Explicit garbage collection after batch
                        logger.debug("Garbage collection run.")

                        # Save checkpoint after successful batch commit
                        current_checkpoint_data = {
                            'version': 1,
                            'current_font_family': family_name,
                            'current_style_name': style_name,
                            'processed_char_offset_in_current_style': i + len(char_batch),
                            'total_records_processed_overall': total_processed_overall,
                            'character_counts': dict(index), # Convert defaultdict to dict for JSON
                            'font_metadata_partial': font_metadata
                        }
                        save_checkpoint(checkpoint_file, current_checkpoint_data)
                        
                        # Periodic progress update
                        current_time = time.time()
                        if current_time - last_update_time > update_interval and total_processed_overall > 0:
                            elapsed = current_time - start_time
                            items_per_second = pbar.n / elapsed if elapsed > 0 else 0 # Use pbar.n for accurate count
                            remaining_items = total_expected_items - pbar.n
                            if items_per_second > 0:
                                estimated_remaining_seconds = remaining_items / items_per_second
                                eta_str = str(timedelta(seconds=int(estimated_remaining_seconds)))
                            else:
                                eta_str = "N/A"
                            
                            logger.info(f"Progress: {pbar.n:,}/{total_expected_items:,} items ({pbar.n/total_expected_items*100:.1f}%)")
                            logger.info(f"Rate: {items_per_second:.2f} items/sec. Elapsed: {timedelta(seconds=int(elapsed))}. ETA: {eta_str}")
                            last_update_time = current_time
                    
                    # After all characters for a style are processed, reset char offset for *next* style
                    # And ensure resume flags are cleared if this was the one being resumed
                    if resume and start_font_family_name == family_name and start_style_name == style_name:
                        processed_char_offset_in_start_style = 0 # Mark this style as done for resume logic
                        start_style_name = None # Move to next style normally

                # After all styles for a font are processed
                if resume and start_font_family_name == family_name:
                     start_font_family_name = None # Move to next font normally
            
            # All processing finished normally
            logger.info("All fonts and styles processed.")
            # Clear checkpoint if processing completed successfully to avoid accidental resume
            if os.path.exists(checkpoint_file):
                logger.info(f"Processing completed successfully. Removing checkpoint file: {checkpoint_file}")
                try:
                    os.remove(checkpoint_file)
                except OSError as e:
                    logger.error(f"Could not remove checkpoint file {checkpoint_file}: {e}")


    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user (KeyboardInterrupt). Last checkpoint should reflect progress.")
        # Checkpoint is saved after each batch. Metadata will be saved in finally.
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
        # Checkpoint is saved after each batch. Metadata will be saved in finally.
    finally:
        logger.info("Attempting to save final metadata...")
        # Final metadata save uses the latest 'index' and 'total_processed_overall'
        # which are updated throughout the process. 'font_metadata' is also built up.
        try:
            with env.begin(write=True) as txn:
                final_metadata = {
                    'total_records': total_processed_overall,
                    'unique_characters': len(index), # index is defaultdict
                    'font_families': font_metadata,
                    'character_counts': dict(index), # Convert to dict for JSON
                    'sources': ['font']
                }
                txn.put(b'__metadata__', json.dumps(final_metadata).encode())
                logger.info(f"Final metadata successfully saved to LMDB. Total records: {total_processed_overall}")
        except Exception as e_meta:
            logger.error(f"Error saving final metadata to LMDB: {e_meta}", exc_info=True)
            # Backup metadata save
            metadata_backup_path = os.path.join(output_dir, 'font_metadata_backup.json')
            try:
                with open(metadata_backup_path, 'w') as f_backup:
                    backup_data = {
                        'total_records': total_processed_overall,
                        'unique_characters': len(index),
                        'font_families': font_metadata,
                        'character_counts': dict(index),
                        'sources': ['font']
                    }
                    json.dump(backup_data, f_backup, ensure_ascii=False, indent=2)
                logger.info(f"Final metadata saved to backup JSON: {metadata_backup_path}")
            except Exception as e_backup_json:
                logger.error(f"Error saving final metadata to backup JSON: {e_backup_json}", exc_info=True)

    logger.info(f"Dataset generation process finished. Total records processed: {total_processed_overall}")
    logger.info(f"Total unique characters in index: {len(index)}")
    
    return env, total_processed_overall


def main(test_mode=False, argv=None):
    """
    Main entry point for the script.
    
    Args:
        test_mode: If True, don't exit on errors (for testing purposes)
        argv: Optional list of arguments to parse (for testing)
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate font-based kanji images with batching and checkpointing.')
    parser.add_argument('--output-dir', type=str, default='/app/output/prep',
                        help='Directory to store output LMDB database and checkpoint file (default: /app/output/prep)')
    parser.add_argument('--limit-fonts', type=int, default=None,
                        help='Limit the number of font families to process (default: use all fonts)')
    parser.add_argument('--source-lmdb', type=str, default='/app/output/prep/kanji.lmdb',
                        help='Path to source LMDB database to get characters from (default: /app/output/prep/kanji.lmdb)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Number of characters to process in a single batch (default: 1000)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume processing from the last checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug level logging')
    parser.add_argument('--memory-warning-threshold', type=float, default=80.0,
                        help='Memory usage percentage (0-100) to trigger a warning (default: 80.0)')
    
    args = parser.parse_args(argv) # Pass argv here

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled.")
    
    # Set up the output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True) # Also for checkpoint file
    
    logger.info("Starting font-based kanji image generation script.")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Source LMDB for characters: {args.source_lmdb}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Memory warning threshold: {args.memory_warning_threshold}%")

    if args.limit_fonts:
        logger.info(f"Limiting to {args.limit_fonts} font families")
    else:
        logger.info("Processing all available font families (or limited by internal list)")
    
    # Note: get_japanese_fonts() is called inside generate_font_kanji_dataset now
    
    logger.info(f"Reading characters from source LMDB database: {args.source_lmdb}")
    characters = get_characters_from_lmdb(args.source_lmdb)
    if not characters:
        logger.error("Failed to get characters from source LMDB database. Exiting.")
        if not test_mode:
            import sys
            sys.exit(1)
        return
    
    logger.info(f"Using {len(characters)} unique characters from source LMDB")
    
    # Generate the dataset
    env, total_processed = generate_font_kanji_dataset(
        output_dir, 
        characters=characters,
        limit_fonts=args.limit_fonts,
        chunk_size=args.chunk_size,
        resume=args.resume,
        checkpoint_path_base=args.output_dir, # Checkpoint in output_dir
        memory_warning_threshold=args.memory_warning_threshold
    )
    
    if env is None: # Indicates a critical failure like no fonts found
        logger.error("Dataset generation failed to initialize (e.g., no fonts). Exiting.")
        if not test_mode:
            import sys
            sys.exit(1)
        return
    
    # Close the LMDB environment
    try:
        env.close()
        logger.info("LMDB environment closed.")
    except Exception as e_close:
        logger.error(f"Error closing LMDB environment: {e_close}", exc_info=True)

    logger.info("\nDataset generation script finished!")
    logger.info(f"Total records processed in this run (or overall if resumed): {total_processed}")
    logger.info(f"LMDB database potentially updated at: {os.path.join(output_dir, 'kanji_fonts.lmdb')}")


if __name__ == "__main__":
    main()
