## Backend
- Language: Python
- Machine Learning: PyTorch
- Image Processing: OpenCV, PIL
- Data Storage: LMDB

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
  - `make test` - Run unit tests in the Docker container
  - `make clean` - Delete all images from output/prep directory
  - `make help` - Display help information

## Image Processing Technologies
- OpenCV for advanced image processing operations:
  - Multiple thresholding methods (Otsu, adaptive, fixed)
  - Morphological operations for stroke connectivity
  - Gaussian blur for noise reduction
- PIL (Python Imaging Library) for basic image operations
- Custom algorithms for touch/stylus input optimization

## Machine Learning Technologies
- PyTorch for model development and training
- CNN architecture for kanji recognition
- Data augmentation for touch input simulation
- Mixed precision training for performance optimization
