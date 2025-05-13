## Project Goals
- [x] Extract modular functions from image processing code
- [x] Create a Makefile for Docker Compose operations
- [/] Improve image processing pipeline
- [ ] Implement kanji recognition model
- [ ] Create a user interface for kanji recognition

## Key Features
- ETL9G dataset processing for kanji images
- Image preprocessing (cropping, padding)
- Docker-based development environment
- Makefile for simplified operations

## Completion Criteria
- Modular, well-documented code
- Accurate kanji recognition
- User-friendly interface
- Comprehensive test coverage

## Completed Tasks
- [2025-05-12] Extracted `extract_item9g_image()` function from `extract_etl9g_images()` for better modularity
- [2025-05-12] Created Makefile with commands for Docker Compose operations (run, shell, build, help)
- [2025-05-12] Implemented `read_records()` generator function to improve modularity of record reading logic
- [2025-05-12] Modified image processing to pad images to 128x128 dimensions for standardization
- [2025-05-13] Added 'clean' command to Makefile to delete all images from output/prep directory
- [2025-05-13] Moved `crop_and_pad()` function from `parse.py` to `clean.py` for better code organization
- [2025-05-13] Created `prepare.py` to implement the ETL pipeline for preparing the ETL9G dataset for ML training

## Completed Tasks (continued)
- [2025-05-13] Updated docker-compose.yml to mount the entire output/prep directory for better data persistence

## In Progress
- Implementing kanji recognition model

## Completed Tasks (continued)
- [2025-05-13] Modified prepare.py to process ETL9G files individually and save to separate pickle files
- [2025-05-13] Added checkpoint mechanism to prepare.py to resume processing from the last file
- [2025-05-13] Added lmdb to requirements.txt and installed liblmdb-dev in Dockerfile for efficient key-value storage
- [2025-05-14] Refactored prepare.py to use LMDB instead of pickle files for more efficient storage of processed kanji data
