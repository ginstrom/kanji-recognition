## Key Components

- **ETL9G Data Processing**: Scripts for processing the ETL9G handwriting database
  - `parse_etl9g.py`: Extracts and processes images from ETL9G data files
  - `clean.py`: Provides image processing functions (cropping, padding, black/white conversion)
  - `datasplit.py`: Splits datasets into train, validation, and test sets
  - `prepare.py`: Prepares the ETL9G dataset for ML training (ETL pipeline)
  - `jis_unicode_map.py`: Maps JIS codes to Unicode characters
  - `genkanji.py`: Generates font-based kanji images using system fonts
    - Now exclusively loads characters from the kanji.lmdb database created by prepare.py
    - Finds Japanese-capable fonts using fc-list
    - Renders characters in regular and bold styles
    - Processes images using the same pipeline as ETL9G data
    - Stores results in an LMDB database similar to prepare.py
    - Supports command-line arguments for configurable font limits
    - Provides detailed progress reporting for long-running processing
    - Includes test mode for unit testing

- **Data Loading and Model Training**: Components for loading data and training models
  - `load.py`: PyTorch Dataset implementation for loading kanji data from LMDB databases
  - `train.py`: Model training pipeline with CNN architecture

- **Touch/Stylus Input Optimization**: Components for optimizing the system for touch/stylus input
  - `convert_to_bw_multi_approach()`: Enhanced B&W conversion algorithm optimized for touch input
  - Touch input simulation for data augmentation (to be implemented)
  - Deployment preprocessing pipeline for real-time touch input (to be implemented)

- **Docker Environment**: Configuration for containerized development
  - `Dockerfile`: Defines the container image
  - `docker-compose.yml`: Configures services and volume mounts
  - `Makefile`: Provides commands for Docker operations

## Data Flow

1. ETL9G data files are read from the `/data/ETL9G/` directory
2. `read_records()` generator function reads fixed-size records from the binary data files
3. For each record, `extract_item9g_image()` extracts:
   - The original image
   - A cropped/padded version of the image
   - The Unicode character
4. `process_kanji_dict()` converts images to binary black and white using the multi-approach method:
   - Multiple thresholding methods (Otsu, adaptive, and fixed)
   - Selection based on appropriate stroke density for kanji
   - Morphological operations to improve stroke connectivity
   - Optimization for touch/stylus input characteristics
5. `prepare.py` orchestrates the entire ETL process:
   - Processes one ETL9G file at a time
   - Stores processed images in an LMDB database with keys in the format `character_index_source`
   - Uses a defaultdict-based index to track character occurrences
   - Stores metadata about the dataset in the LMDB database
   - Saves the LMDB database to the output/prep directory

6. The processed ETL9G dataset in the LMDB database contains:
   - 607,200 total records
   - 3,036 unique kanji characters
   - 200 images per character (consistent across all characters)
   - Approximately 11.61 GB of data on disk

7. For model training:
   - Images are loaded from the LMDB database using the `KanjiLMDBDataset` class
   - Data is augmented to simulate touch input variations
   - Model is trained on binary images to match deployment conditions
   - Evaluation metrics track performance on touch-like input

## Project Structure

```
kanji-recognition/
├── cline_docs/             # Documentation for development
├── data/                   # Data directory
│   └── ETL9G/              # ETL9G dataset files
├── docs/                   # Project documentation
│   ├── data.md             # Dataset documentation
│   └── outline.md          # Project outline and architecture
├── output/                 # Output directory
│   └── prep/               # Prepared data for ML training
│       └── etl9g_images/   # Processed images for viewing
├── src/                    # Source code
│   ├── clean.py            # Image processing functions
│   ├── datasplit.py        # Dataset splitting functions
│   ├── jis_unicode_map.py  # JIS to Unicode mapping
│   ├── lmdb_stats.py       # LMDB database statistics
│   ├── load.py             # PyTorch Dataset for loading LMDB data
│   ├── parse_etl9g.py      # ETL9G data extraction
│   ├── prepare.py          # ETL pipeline for dataset preparation
│   ├── requirements.txt    # Python dependencies
│   └── train.py            # Model training
├── tests/                  # Unit tests
│   ├── conftest.py         # Shared pytest fixtures
│   ├── test_clean.py       # Tests for clean.py
│   ├── test_datasplit.py   # Tests for datasplit.py
│   ├── test_genkanji.py    # Tests for genkanji.py
│   ├── test_load.py        # Tests for load.py
│   ├── test_parse_etl9g.py # Tests for parse_etl9g.py
│   └── test_prepare.py     # Tests for prepare.py
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Docker image definition
├── Makefile                # Build automation
└── README.md               # Project overview
```

## Recent Significant Changes

- [2025-05-17] Fixed LMDB environment handling in genkanji.py to properly close environments and avoid "closed/deleted/dropped object" errors
- [2025-05-17] Added test_mode parameter to main() function in genkanji.py to support unit testing
- [2025-05-17] Updated genkanji.py to exclusively load characters from kanji.lmdb database
- [2025-05-17] Modified genkanji.py to generate images only for characters in the existing kanji.lmdb database
- [2025-05-17] Modified genkanji.py to generate images for all Japanese kanji characters with improved progress reporting
- [2025-05-17] Added command-line arguments to make font and character limits configurable
- [2025-05-17] Added genkanji.py module to generate font-based kanji images for all hiragana and kanji characters
- [2025-05-17] Implemented comprehensive unit tests for genkanji.py module
- [2025-05-17] Added fonttools dependency for font detection and processing
- [2025-05-16] Updated project documentation to focus on touch/stylus input optimization
- [2025-05-16] Documented the improved B&W conversion algorithm for touch input
- [2025-05-16] Reprioritized roadmap to focus on touch input optimization
- [2025-05-16] Implemented basic kanji recognition model with CNN architecture
- [2025-05-16] Created training pipeline with cross-entropy loss and Adam optimizer
- [2025-05-15] Created load.py module with PyTorch Dataset implementation for loading kanji data from LMDB databases
- [2025-05-15] Implemented comprehensive unit tests for load.py module
- [2025-05-15] Fixed bug in datasplit.py to handle different character count keys in metadata ('character_counts' vs 'character_counts_etl9g')
- [2025-05-15] Refactored datasplit.py to use LMDB databases for train, validation, and test splits with configurable ratios
- [2025-05-14] Refactored prepare.py to use LMDB instead of pickle files for storing processed kanji data
- [2025-05-14] Modified parse.py to no longer write PNG files to disk, improving efficiency by directly passing image data to prepare.py for LMDB storage
- [2025-05-14] Created `lmdb_stats.py` script to read metadata from the LMDB database and provide comprehensive statistics about the kanji dataset
- [2025-05-14] Renamed `parse.py` to `parse_etl9g.py` to better reflect its specific purpose in processing the ETL9G dataset
- [2025-05-14] Added pytest and related packages to requirements.txt for unit testing
- [2025-05-14] Created comprehensive unit tests for parse_etl9g.py, clean.py, and prepare.py modules
- [2025-05-13] Modified prepare.py to process ETL9G files individually and save to separate pickle files
- [2025-05-13] Added checkpoint mechanism to prepare.py to resume processing from the last file
- [2025-05-13] Added lmdb to requirements.txt and installed liblmdb-dev in Dockerfile for efficient key-value storage
- [2025-05-13] Created `prepare.py` to implement the ETL pipeline for preparing the ETL9G dataset for ML training
- [2025-05-13] Moved `crop_and_pad()` function from `parse.py` to `clean.py` for better code organization
- [2025-05-13] Added 'clean' command to Makefile to delete all images from output/prep directory
- [2025-05-12] Modified image processing to standardize images to 128x128 dimensions
- [2025-05-12] Implemented `read_records()` generator function to improve modularity of record reading logic
- [2025-05-12] Created Makefile with commands for Docker Compose operations
- [2025-05-12] Extracted `extract_item9g_image()` function from `extract_etl9g_images()` for better modularity

## Development Workflow

1. Use `make run CMD` to execute commands in the Docker container
2. Use `make shell` to open an interactive shell in the container
3. All Python scripts should be run within the Docker environment
4. To prepare the ETL9G dataset for ML training:
   ```
   make run python prepare.py
   ```
5. To view statistics about the processed kanji dataset:
   ```
   make run python lmdb_stats.py
   ```
   Optional arguments:
   - `--top-n N`: Display top N most common characters (default: 10)
   - `--bottom-n N`: Display bottom N least common characters (default: 10)
   - `--sample-images N`: Display N sample images as ASCII art (default: 0)
   - `--visualize`: Generate and save visualization of character distribution
   - `--verbose`: Display more detailed information
6. To run the unit tests:
   ```
   make test
   ```
   Optional arguments (passed via TEST_ARGS):
   ```
   # Verbose output
   make test TEST_ARGS="-v"
   
   # Generate coverage report
   make test TEST_ARGS="--cov=src"
   
   # Run specific tests by name
   make test TEST_ARGS="-k test_name"
   
   # Run tests in a specific file
   make test TEST_ARGS="tests/test_file.py"
   
   # Combine multiple options
   make test TEST_ARGS="-v --cov=src tests/test_clean.py"
   ```
   
   Note: Always use the TEST_ARGS variable to pass arguments to pytest. Do not pass arguments directly to the make test command.
7. To split the kanji dataset into train, validation, and test sets:
   ```
   make run python datasplit.py
   ```
   Optional arguments:
   - `--source PATH`: Path to source LMDB database (default: /app/output/prep/kanji.lmdb)
   - `--output-dir DIR`: Directory to store output LMDB databases (default: /app/output/prep)
   - `--train-ratio RATIO`: Fraction of data to use for training (default: 0.8)
   - `--val-ratio RATIO`: Fraction of data to use for validation (default: 0.1)
   - `--random-seed SEED`: Random seed for reproducibility (default: 42)
   
   Example with custom ratios:
   ```
   make run python datasplit.py --train-ratio 0.7 --val-ratio 0.15
   ```

8. To load and visualize the kanji dataset for model training:
   ```
   make run python load.py --visualize
   ```
   Optional arguments:
   - `--data-dir DIR`: Directory containing the LMDB databases (default: /app/output/prep)
   - `--batch-size N`: Batch size for DataLoader (default: 32)
   - `--num-workers N`: Number of worker processes for DataLoader (default: 4)
   - `--cache-size N`: Number of samples to cache in memory (default: 0)
   - `--visualize`: Visualize random samples from the datasets
   - `--num-samples N`: Number of samples to visualize (default: 9)

9. To train the kanji recognition model:
   ```
   make run python train.py
   ```
   Optional arguments:
   - `--data-dir DIR`: Directory containing the LMDB databases (default: /app/output/prep)
   - `--batch-size N`: Batch size for DataLoader (default: 32)
   - `--num-workers N`: Number of worker processes for DataLoader (default: 4)
   - `--epochs N`: Number of training epochs (default: 10)
   - `--learning-rate LR`: Learning rate for optimizer (default: 0.001)
   - `--model-dir DIR`: Directory to save trained models (default: /app/output/models)

10. To generate font-based kanji images:
   ```
   make run python genkanji.py
   ```
   Optional arguments:
   - `--output-dir DIR`: Directory to store output LMDB database (default: /app/output/prep)
   - `--limit-fonts N`: Limit the number of font families to process (default: None)
   - `--source-lmdb PATH`: Path to source LMDB database to get characters from (default: /app/output/prep/kanji.lmdb)
   
   Examples:
   ```
   # Process all characters from kanji.lmdb with 3 font families
   make run python genkanji.py --limit-fonts 3
   
   # Process all characters from kanji.lmdb with all available fonts
   make run python genkanji.py
   
   # Process characters from a different LMDB database
   make run python genkanji.py --source-lmdb /path/to/other.lmdb
   ```

11. Output files:
   - LMDB database with processed ETL9G data is saved to the `output/prep/kanji.lmdb` directory
   - LMDB database with font-generated kanji images is saved to the `output/prep/kanji_fonts.lmdb` directory
   - Train split is saved to the `output/prep/kanji.train.lmdb` directory
   - Validation split is saved to the `output/prep/kanji.val.lmdb` directory
   - Test split is saved to the `output/prep/kanji.test.lmdb` directory
   - Character distribution visualization (if generated) is saved to `output/prep/char_distribution.png`
   - Trained models are saved to the `output/models` directory
