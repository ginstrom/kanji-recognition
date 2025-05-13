## Current Objective
Modify prepare.py to process ETL9G files individually and implement checkpoint mechanism

## Context
The prepare.py script is getting killed when processing all ETL9G files at once and storing them in a single pickle file. This is likely due to memory constraints. We need to modify the script to process one file at a time, save each to a separate pickle file, and implement a checkpoint mechanism to resume processing from the last successfully processed file.

## Implementation Plan
1. Modify the `prepare_etl9g_dataset` function to:
   - Process one ETL9G file at a time
   - Save each file's processed data to a separate pickle file (e.g., `etl9g_data_01.pkl`)
   - Return a list of all generated pickle files

2. Add a checkpoint mechanism to:
   - Check if a pickle file for a given ETL9G file already exists
   - Skip processing if the file exists, otherwise process it
   - Keep track of the last successfully processed file

3. Update the `split_and_save_datasets` function to:
   - Accept a list of pickle files instead of a single file
   - Load data from all pickle files and combine them for splitting

4. Update the `main` function to implement the new workflow

## Next Steps
1. Test the modified script by running it in the Docker container
2. Verify that it can process all ETL9G files without getting killed
3. Verify that it can resume processing from the last successfully processed file
4. Verify that the train/validation/test splits are correctly generated
