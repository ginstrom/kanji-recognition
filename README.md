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
- `make help` - Display help information

## Data

See [docs/data.md](docs/data.md)