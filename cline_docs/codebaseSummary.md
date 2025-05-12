## Key Components

- **ETL9G Data Processing**: Scripts for processing the ETL9G handwriting database
  - `parse_sample.py`: Extracts and processes images from ETL9G data files
  - `jis_unicode_map.py`: Maps JIS codes to Unicode characters

- **Docker Environment**: Configuration for containerized development
  - `Dockerfile`: Defines the container image
  - `docker-compose.yml`: Configures services and volume mounts
  - `Makefile`: Provides commands for Docker operations

## Data Flow

1. ETL9G data files are read from the `/data/ETL9G/` directory
2. `read_records()` generator function reads fixed-size records from the binary data files
3. `extract_etl9g_images()` processes these records using the generator
4. For each record, `extract_item9g_image()` extracts:
   - The original image
   - A cropped/padded version of the image
   - The Unicode character
5. Images are saved to the output directory for viewing

## Project Structure

```
kanji-recognition/
├── cline_docs/             # Documentation for development
├── data/                   # Data directory
│   └── ETL9G/              # ETL9G dataset files
├── docs/                   # Project documentation
├── etl9g_images/           # Output directory for processed images
├── src/                    # Source code
│   ├── jis_unicode_map.py  # JIS to Unicode mapping
│   ├── parse_sample.py     # ETL9G data processing
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

## Development Workflow

1. Use `make run CMD` to execute commands in the Docker container
2. Use `make shell` to open an interactive shell in the container
3. All Python scripts should be run within the Docker environment
4. Output images are saved to the `etl9g_images` directory
