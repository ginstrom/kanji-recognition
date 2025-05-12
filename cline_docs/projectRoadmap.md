## Project Goals
- [x] Extract modular functions from image processing code
- [x] Create a Makefile for Docker Compose operations
- [ ] Improve image processing pipeline
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

## In Progress
- Improving Docker workflow and development experience
