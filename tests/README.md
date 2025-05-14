# Unit Tests for Kanji Recognition Project

This directory contains unit tests for the kanji recognition project using pytest.

## Test Structure

- `conftest.py`: Shared pytest fixtures used across all test files
- `test_parse_etl9g.py`: Tests for ETL9G parsing functions
- `test_clean.py`: Tests for image processing functions
- `test_prepare.py`: Tests for LMDB storage and ETL pipeline

## Running Tests

Run all tests from the project root directory:

```bash
make run python -m pytest
```

Run tests with verbose output:

```bash
make run python -m pytest -v
```

Run tests with coverage report:

```bash
make run python -m pytest --cov=src
```

Run specific tests:

```bash
make run python -m pytest tests/test_parse_etl9g.py
make run python -m pytest -k "test_read_records"
```

## Test Design

The tests use a combination of:

1. **Mock objects**: Using `unittest.mock` to isolate components
2. **Fixtures**: Shared test data and setup/teardown logic
3. **Parameterization**: Testing multiple inputs with the same test logic
4. **Temporary directories**: For testing file I/O operations

## Test Coverage

The tests cover:

- ETL9G record parsing and image extraction
- Image processing (cropping, padding, black/white conversion)
- LMDB database operations
- Error handling and edge cases

## Adding New Tests

When adding new tests:

1. Follow the existing naming conventions
2. Use appropriate fixtures from `conftest.py`
3. Mock external dependencies
4. Test both normal operation and error handling
