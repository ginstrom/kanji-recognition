## Current Objective
Modify the `genkanji.py` script to always load from the kanji.lmdb database created by `src/prepare.py` and remove all other code to write all characters.

## Context
This task is part of refining the data generation pipeline for the kanji recognition system. The goal is to ensure that the font-generated images match exactly the characters that were extracted from the handwriting database, maintaining consistency in the training data.

## Completed Steps
1. Updated `genkanji.py` to:
   - Always load characters from the kanji.lmdb database
   - Fixed an issue with the LMDB environment being closed prematurely
   - Modified the `get_characters_from_lmdb` function to properly handle the LMDB environment
   - Added a `test_mode` parameter to the `main()` function to support testing

2. Updated the tests to work with the modified script:
   - Fixed the `test_get_characters_from_lmdb` test to properly mock the JSON loading
   - Updated the `test_main_no_fonts` and `test_main_with_fonts` tests to use the new `test_mode` parameter

3. Verified that all tests pass and the script works correctly with the LMDB database.

## Next Steps
1. Consider adding more error handling for edge cases
2. Optimize the font generation process for better performance
3. Add more documentation about the LMDB database format and usage
