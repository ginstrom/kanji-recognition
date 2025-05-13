## Current Objective
Refactor the prepare.py script to use LMDB instead of pickle files

## Context
We previously added lmdb to the requirements.txt file and installed liblmdb-dev in the Docker container. Now we're refactoring the prepare.py script to use LMDB for storing processed kanji data instead of pickle files.

## Implementation Plan
1. Replace the existing pickle-based storage with LMDB
2. Create an index dictionary (defaultdict) to track character occurrences
3. Store each character with a unique key: `character_index_source`
4. Store the black and white image bytes directly in LMDB
5. Remove the dataset splitting functionality (to be reimplemented later)
6. Ensure the LMDB database is stored in the output/prep directory

## Next Steps
1. Implement the refactored prepare.py script
2. Test the script to ensure it correctly processes ETL9G files and stores data in LMDB
3. Verify that the LMDB database is accessible from the Docker container
