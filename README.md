# kanji-recognition

ML model for recognizing handwritten kanji characters

## Data Processing Pipeline

This project includes a complete data processing pipeline for the ETL9G dataset:

1. **Parsing**: Extract images and metadata from ETL9G binary files
   ```
   make run python parse_etl9g.py
   ```

2. **Preparation**: Process images and store in LMDB database
   ```
   make run python prepare.py
   ```

3. **Dataset Splitting**: Split the dataset into train, validation, and test sets
   ```
   make run python datasplit.py
   ```
   
   Optional arguments:
   - `--source PATH`: Path to source LMDB database (default: /app/output/prep/kanji.lmdb)
   - `--output-dir DIR`: Directory to store output LMDB databases (default: /app/output/prep)
   - `--train-ratio RATIO`: Fraction of data to use for training (default: 0.8)
   - `--val-ratio RATIO`: Fraction of data to use for validation (default: 0.1)
   - `--random-seed SEED`: Random seed for reproducibility (default: 42)
   
   Example with custom ratios:
   ```
   make run python datasplit.py --train-ratio 0.7 --val-ratio 0.15
   ```

## Development Setup

This project uses Docker and Docker Compose for development to ensure a consistent environment.

### Prerequisites

- Docker
- Docker Compose
- Make

### Getting Started

1. Clone the repository
2. Build the Docker container:
   ```
   make build
   ```
3. Run commands in the Docker environment:
   ```
   make run CMD
   ```
   For example:
   ```
   make run python parse_sample.py
   ```
4. Open a shell in the Docker environment:
   ```
   make shell
   ```

### Available Make Commands

- `make run CMD` - Run a command in the Docker container
- `make shell` - Open a shell in the Docker container
- `make build` - Build the Docker container
- `make test` - Run unit tests in the Docker container
- `make clean` - Delete all images from output/prep directory
- `make help` - Display help information

### Running Tests

Run all tests:
```
make test
```

Run tests with additional options:
```
# Verbose output
make test TEST_ARGS="-v"

# Generate coverage report
make test TEST_ARGS="--cov=src"

# Run specific tests by name
make test TEST_ARGS="-k test_name"

# Run tests in a specific file
make test TEST_ARGS="tests/test_file.py"

# Combine multiple options
make test TEST_ARGS="-v --cov=src tests/test_clean.py"
```

Note: Always use the TEST_ARGS variable to pass arguments to pytest. Do not pass arguments directly to the make test command.

## Data

See [docs/data.md](docs/data.md)
