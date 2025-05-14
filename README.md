# kanji-recognition

ML model for recognizing handwritten kanji characters

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
