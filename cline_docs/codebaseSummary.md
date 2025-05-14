## Key Components

- **ETL9G Data Processing**: Scripts for processing the ETL9G handwriting database
  - `parse_etl9g.py`: Extracts and processes images from ETL9G data files
  - `clean.py`: Provides image processing functions (cropping, padding, black/white conversion)
  - `datasplit.py`: Splits datasets into train, validation, and test sets
  - `prepare.py`: Prepares the ETL9G dataset for ML training (ETL pipeline)
  - `jis_unicode_map.py`: Maps JIS codes to Unicode characters

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
4. `process_kanji_dict()` converts images to two-bit black and white
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

## Project Structure

```
kanji-recognition/
├── cline_docs/             # Documentation for development
├── data/                   # Data directory
│   └── ETL9G/              # ETL9G dataset files
├── docs/                   # Project documentation
├── output/                 # Output directory
│   └── prep/               # Prepared data for ML training
│       └── etl9g_images/   # Processed images for viewing
├── src/                    # Source code
│   ├── clean.py            # Image processing functions
│   ├── datasplit.py        # Dataset splitting functions
│   ├── jis_unicode_map.py  # JIS to Unicode mapping
│   ├── lmdb_stats.py       # LMDB database statistics
│   ├── parse_etl9g.py      # ETL9G data extraction
│   ├── prepare.py          # ETL pipeline for dataset preparation
│   ├── requirements.txt    # Python dependencies
│   └── train.py            # Model training (to be implemented)
├── tests/                  # Unit tests
│   ├── conftest.py         # Shared pytest fixtures
│   ├── test_clean.py       # Tests for clean.py
│   ├── test_parse_etl9g.py # Tests for parse_etl9g.py
│   └── test_prepare.py     # Tests for prepare.py
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Docker image definition
├── Makefile                # Build automation
└── README.md               # Project overview
```

## Recent Significant Changes

- [2025-05-12] Extracted `extract_item9g_image()` function from `extract_etl9g_images()` for better modularity
- [2025-05-12] Created Makefile with commands for Docker Compose operations
- [2025-05-12] Implemented `read_records()` generator function to improve modularity of record reading logic
- [2025-05-12] Modified image processing to standardize images to 128x128 dimensions
- [2025-05-13] Added 'clean' command to Makefile to delete all images from output/prep directory
- [2025-05-13] Moved `crop_and_pad()` function from `parse.py` to `clean.py` for better code organization
- [2025-05-13] Created `prepare.py` to implement the ETL pipeline for preparing the ETL9G dataset for ML training
- [2025-05-13] Updated docker-compose.yml to mount the entire output/prep directory for better data persistence
- [2025-05-13] Modified parse.py to use the correct output directory path that matches the Docker volume mount
- [2025-05-13] Modified prepare.py to process ETL9G files individually and save to separate pickle files
- [2025-05-13] Added checkpoint mechanism to prepare.py to resume processing from the last file
- [2025-05-13] Added lmdb to requirements.txt and installed liblmdb-dev in Dockerfile for efficient key-value storage
- [2025-05-14] Refactored prepare.py to use LMDB instead of pickle files for storing processed kanji data
- [2025-05-14] Modified parse.py to no longer write PNG files to disk, improving efficiency by directly passing image data to prepare.py for LMDB storage
- [2025-05-14] Created `lmdb_stats.py` script to read metadata from the LMDB database and provide comprehensive statistics about the kanji dataset
- [2025-05-14] Renamed `parse.py` to `parse_etl9g.py` to better reflect its specific purpose in processing the ETL9G dataset
- [2025-05-14] Added pytest and related packages to requirements.txt for unit testing
- [2025-05-14] Created comprehensive unit tests for parse_etl9g.py, clean.py, and prepare.py modules

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
7. Output files:
   - LMDB database with processed data is saved to the `output/prep/kanji.lmdb` directory
   - Character distribution visualization (if generated) is saved to `output/prep/char_distribution.png`
