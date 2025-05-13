## Key Components

- **ETL9G Data Processing**: Scripts for processing the ETL9G handwriting database
  - `parse.py`: Extracts and processes images from ETL9G data files
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
   - Stores each file's processed images in a separate pickle file (e.g., `etl9g_data_01.pkl`)
   - Implements a checkpoint mechanism to resume processing from the last successfully processed file
   - Loads data from all pickle files for splitting
   - Splits the data into train, validation, and test sets using `split_dataset()`
   - Saves the split datasets to separate pickle files in the output/prep directory

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
│   ├── parse.py            # ETL9G data extraction
│   ├── prepare.py          # ETL pipeline for dataset preparation
│   ├── requirements.txt    # Python dependencies
│   └── train.py            # Model training (to be implemented)
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

## Development Workflow

1. Use `make run CMD` to execute commands in the Docker container
2. Use `make shell` to open an interactive shell in the container
3. All Python scripts should be run within the Docker environment
4. To prepare the ETL9G dataset for ML training:
   ```
   make run python prepare.py
   ```
5. Output files:
   - Processed images are saved to the `output/prep/etl9g_images` directory
   - Pickle files with processed data are saved to the `output/prep` directory:
     - `etl9g_data_XX.pkl`: Processed data for each ETL9G file (where XX is the file number)
     - `etl9g_train.pkl`: Training set
     - `etl9g_val.pkl`: Validation set
     - `etl9g_test.pkl`: Test set
