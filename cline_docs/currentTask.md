## Current Objective
Create a script in the src directory that reads metadata from the LMDB database and provides statistics about the kanji dataset.

## Context
The ETL9G dataset has been processed and stored in an LMDB database at `/app/output/prep/kanji.lmdb`. The database contains processed kanji images and metadata. We need a script that can read this database, extract the metadata, and provide useful statistics about the dataset.

## Implementation Plan
1. Create a new Python script `lmdb_stats.py` in the src directory
2. Implement functionality to open and read from the LMDB database
3. Extract the metadata stored with the key `__metadata__`
4. Calculate additional statistics by analyzing the database contents
5. Display the statistics in a clear, formatted way

## Statistics to Include
- Basic metadata (from `__metadata__` key):
  - Total number of records
  - Number of unique characters
  - Character counts per character
- Additional calculated statistics:
  - Most common characters (top N with counts)
  - Least common characters (bottom N with counts)
  - Distribution of character frequencies
  - Average images per character
  - Database size on disk
  - Sample key format and structure

## Next Steps
1. Create the `lmdb_stats.py` script with the functionality described above
2. Test the script to ensure it correctly reads and displays statistics from the LMDB database
3. Document the script's usage in the project documentation
