# Project Data

## ETL9G Handwritten Japanese Database

- num characters: 3,036 (2,965 kanjis, 71 hiraganas)
- num writers: 4,000
- num samples: 607,200
- Download at [http://etlcdb.db.aist.go.jp/download2/](http://etlcdb.db.aist.go.jp/download2/)

### Data characteristics

Binary format, with each character data occupying 8199 bytes.

The first portion is the header, which among other data contains the JIS character code of the character.

The final 8128 bytes store the 127x1287 pixel image of the handwritten character. Due to the way the characters were scanned, there is often some bleed-over from the adjacent character. To eliminate this issue, the preparation phase crops each image to the center 70%, padding the cropped image to get a 128x128 image.

The image is then converted from grayscale to black and white (binary).

## Data Processing Pipeline

### 1. Parsing (parse_etl9g.py)
- Extracts images and metadata from ETL9G binary files
- Converts JIS character codes to Unicode
- Crops and pads images to 128x128 pixels

### 2. Preparation (prepare.py)
- Processes images and stores them in an LMDB database
- Stores metadata about the dataset in the LMDB database
- Uses a key format of `character_index_source` for each image
- Tracks character occurrences in a defaultdict-based index

### 3. Dataset Splitting (datasplit.py)
- Splits the LMDB database into train, validation, and test sets
- Uses stratified sampling to ensure each character appears in each split
- Creates separate LMDB databases for each split
- Stores metadata about each split in its respective LMDB database
- Configurable split ratios (default: 80% train, 10% validation, 10% test)
- Ensures reproducibility with a configurable random seed

The resulting LMDB databases are stored in the following locations:
- Source database: `output/prep/kanji.lmdb`
- Train split: `output/prep/kanji.train.lmdb`
- Validation split: `output/prep/kanji.val.lmdb`
- Test split: `output/prep/kanji.test.lmdb`

Each split database contains the images and their corresponding character labels, stored in a format compatible with PyTorch Dataset classes.
