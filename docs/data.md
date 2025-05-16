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

The image is then converted from grayscale to black and white (binary) using an enhanced multi-approach algorithm optimized for touch/stylus input.

## Touch/Stylus Input Optimization

### Enhanced B&W Conversion Algorithm

The project uses an improved black and white conversion algorithm (`convert_to_bw_multi_approach()`) specifically optimized for touch/stylus input:

1. **Multiple Thresholding Methods**:
   - Otsu's thresholding: Automatically determines optimal threshold value
   - Adaptive thresholding: Adjusts threshold based on local image regions
   - Fixed value thresholding: Uses a predefined threshold value

2. **Intelligent Method Selection**:
   - Analyzes stroke density to select the most appropriate method
   - Targets 10-40% foreground pixel ratio, typical for kanji characters
   - Ensures consistent stroke representation regardless of input variations

3. **Stroke Enhancement**:
   - Applies morphological operations to improve stroke connectivity
   - Enhances thin strokes that are common in touch/stylus input
   - Preserves stroke junctions and endpoints

4. **Noise Reduction**:
   - Applies Gaussian blur to reduce noise before thresholding
   - Removes artifacts and speckles common in touch input

### Touch Input Simulation for Training

To ensure the model generalizes well to real touch/stylus input, the training process includes:

1. **Stroke Thickness Variation**: Simulates different pen/stylus pressure
2. **Jitter Simulation**: Adds realistic hand tremor and touch screen inaccuracies
3. **Stroke Connectivity Variations**: Simulates breaks in strokes that can occur with touch input
4. **Input Device Characteristics**: Mimics the specific characteristics of touch screens and stylus devices

### Real-time Processing for Deployment

The preprocessing pipeline for deployment is optimized for:

1. **Low Latency**: Efficient processing for real-time user experience
2. **Consistent Results**: Standardized input regardless of device variations
3. **Robustness**: Handles various touch input styles and qualities

## Data Processing Pipeline

### 1. Parsing (parse_etl9g.py)
- Extracts images and metadata from ETL9G binary files
- Converts JIS character codes to Unicode
- Crops and pads images to 128x128 pixels

### 2. Preparation (prepare.py)
- Processes images and stores them in an LMDB database
- Applies the enhanced B&W conversion algorithm optimized for touch input
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

### 4. Training with Touch Input Simulation
- Applies augmentation techniques to simulate touch input variations
- Trains on binary images to match deployment conditions
- Focuses on robustness to different touch input styles

The resulting LMDB databases are stored in the following locations:
- Source database: `output/prep/kanji.lmdb`
- Train split: `output/prep/kanji.train.lmdb`
- Validation split: `output/prep/kanji.val.lmdb`
- Test split: `output/prep/kanji.test.lmdb`

Each split database contains the images and their corresponding character labels, stored in a format compatible with PyTorch Dataset classes.

## Testing with Real Touch Input

To evaluate the system's performance with real touch/stylus input:

1. **User Testing Interface**: A simple demo interface for collecting real touch input samples
2. **Comparison Metrics**: Methods for comparing performance between different preprocessing approaches
3. **Error Analysis**: Tools for identifying and addressing issues specific to touch input
4. **Iterative Improvement**: Process for incorporating user feedback to refine the system
