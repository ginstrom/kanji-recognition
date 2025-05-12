## Backend
- Language: Python

## DevOps
- Containerization: Docker
- Execution Environment: Docker Compose
- Build Automation: Makefile

## Development Workflow
- All Python scripts should be executed within the Docker Compose environment to ensure consistency and manage dependencies.
- Use the Makefile for common operations:
  - `make run CMD` - Run a command in the Docker container
  - `make shell` - Open a shell in the Docker container
  - `make build` - Build the Docker container
  - `make help` - Display help information
