# Makefile for kanji-recognition project

# Variables
DOCKER_COMPOSE = docker-compose
SERVICE_NAME = handwriting

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make run CMD    - Run CMD in the Docker container (e.g., make run ls)"
	@echo "  make shell      - Open a shell in the Docker container"
	@echo "  make build      - Build the Docker container"
	@echo "  make test       - Run unit tests in the Docker container"
	@echo "  make clean      - Delete all images from output/prep directory"
	@echo "  make help       - Display this help message"

# Run a command in the Docker container
.PHONY: run
run:
	@if [ -z "$(filter-out run,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make run CMD"; \
		echo "Example: make run ls"; \
		exit 1; \
	fi
	$(DOCKER_COMPOSE) run --rm $(SERVICE_NAME) $(filter-out run,$(MAKECMDGOALS))

# Open a shell in the Docker container
.PHONY: shell
shell:
	$(DOCKER_COMPOSE) run --rm $(SERVICE_NAME) /bin/bash

# Build the Docker container
.PHONY: build
build:
	$(DOCKER_COMPOSE) build

# Run unit tests in the Docker container
.PHONY: test
test:
	$(DOCKER_COMPOSE) run --rm $(SERVICE_NAME) /bin/bash -c "cd /app && python -m pytest $(TEST_ARGS)"

# Clean the output/prep directory
.PHONY: clean
clean:
	$(DOCKER_COMPOSE) run --rm $(SERVICE_NAME) rm -rf /app/output/prep/*

# This allows passing arguments to the run command
%:
	@:
